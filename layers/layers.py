"""Euclidean layers."""
import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act(config, num_layers, enc=True):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    model_config = config.model
    act = getattr(nn, model_config.act)
    acts = [act()] * (num_layers)

    if enc:
        dims = [model_config.hidden_dim] * (num_layers+1) # len=args.num_layers+1
    else:
        dims = [model_config.dim]+[model_config.hidden_dim] * (num_layers)   # len=args.num_layers+1

    return dims, acts


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class GCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0., act=nn.ReLU(), edge_dim=0, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.att = DenseAtt(out_dim, dropout, edge_dim=edge_dim)
        self.msg_transform = msg_transform
        self.sum_transform = sum_transform
        self.act = act
        if msg_transform:
            self.msg_net = nn.Sequential(
                nn.Linear(out_dim+1, out_dim),
                act,
                nn.Linear(out_dim, out_dim)
            )
        if sum_transform:
            self.out_net = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                act,
                nn.Linear(out_dim, out_dim)
            )
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, input):
        x, adj = input

        x = self.linear(x)
        x = self.Agg(x, adj)
        x = self.ln(x)
        x = self.act(x)
        return x, adj

    def Agg(self, x, adj):
        b, n, _ = x.size()
        # b x n x 1 x d     0,0,...0,1,1,...1...
        x_left = torch.unsqueeze(x, 2)
        x_left = x_left.expand(-1, -1, n, -1)
        # b x 1 x n x d     0,1,...n-1,0,1,...n-1...
        x_right = torch.unsqueeze(x, 1)
        x_right = x_right.expand(-1, n, -1, -1)

        if self.msg_transform:
            x_right_ = self.msg_net(torch.cat([x_right,adj.unsqueeze(-1)],dim=-1))
        else:
            x_right_ = x
        if self.edge_dim > 0:
            edge_attr = adj.unsqueeze(-1)
        else:
            edge_attr = None
        att = self.att(x_left, x_right,adj,edge_attr)  # (b*n_node*n_node,dim)
        msg = x_right_ * att
        msg = torch.sum(msg,dim=2)
        if self.sum_transform:
            msg = self.out_net(msg)
        x = x + msg
        return x


class GATLayer(nn.Module):


    def __init__(self, in_dim, out_dim, dropout=0., act=nn.LeakyReLU(0.5), edge_dim=0, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln',num_of_heads=4):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_proj = nn.Linear(in_dim, out_dim,bias=False)
        self.scoring_fn = nn.Linear(2*out_dim//num_of_heads+1,1,bias=False)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.num_of_heads = num_of_heads
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.skip_proj = None
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        x, adj = input

        b, n, _ = x.size()
        x = self.dropout(x)
        nodes_features_proj = self.linear_proj(x).view(b,n,self.num_of_heads,-1)  # (b,n,n_head,dim_out/n_head)
        nodes_features_proj = self.dropout(nodes_features_proj)
        x_left = torch.unsqueeze(nodes_features_proj, 2)
        x_left = x_left.expand(-1, -1, n, -1, -1)
        x_right = torch.unsqueeze(nodes_features_proj, 1)
        x_right = x_right.expand(-1, n, -1, -1, -1)  # (b,n,n,n_head,dim_out/n_head)
        score = self.scoring_fn(torch.cat([x_left,x_right,adj[...,None,None].expand(-1,-1,-1,self.num_of_heads,-1)],dim=-1)).squeeze()
        score = self.leakyReLU(score)  # (b,n,n,n_head)
        edge_mask = (adj > 1e-5).float()
        pad_mask = 1 - edge_mask
        zero_vec = -9e15 * pad_mask  # (b,n,n)

        att = score + zero_vec.unsqueeze(-1).expand(-1, -1,-1, self.num_of_heads)  # (b,n,n,n_head) padding的地方会-9e15
        att = torch.softmax(att,dim=2).transpose(2,3)  # (b,n,n_head,n)
        att = self.dropout(att).transpose(2, 3).unsqueeze(-1)
        msg = x_right * att  # (b,n,n,n_head,dim_out/n_head)
        msg = torch.sum(msg,dim=2)  # (b,n,n_head,dim_out/n_head)
        if self.in_dim != self.out_dim:
            x = self.skip_proj(x)  # (b,n,dim_out)

        x = x+msg.view(b,n,-1)+self.bias  # (b,n,dim_out)
        # x = self.ln(x)
        x = self.act(x)
        return x,adj

'''
InnerProductDecdoer implemntation from:
https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py
'''


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout=0, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, emb_in, emb_out):
        cos_dist = emb_in * emb_out
        probs = self.act(cos_dist.sum(1))
        return probs
