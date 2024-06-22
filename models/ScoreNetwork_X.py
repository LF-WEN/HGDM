import torch
import torch.nn.functional as F
from geoopt import PoincareBall
from torch import nn

from layers.hyp_layers import HGCLayer, HGATLayer
from layers.layers import GCLayer, GATLayer
from manifolds.utils import exp_after_transp0
from models.layers import DenseGCNConv, MLP
from models.utils import get_timestep_embedding
from utils.graph_utils import mask_x, pow_tensor
from models.attention import AttentionLayer


class ScoreNetworkX(torch.nn.Module):

    def __init__(self, max_feat_num, depth, nhid):

        super(ScoreNetworkX, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat, 
                            use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags,t):
        # x = x + t
        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)

        return x

class ScoreNetworkX_poincare(torch.nn.Module):

    def __init__(self, max_feat_num, depth, nhid,manifold,edge_dim,GCN_type,**kwargs):

        super(ScoreNetworkX_poincare, self).__init__()
        self.manifold = manifold
        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid
        if GCN_type == 'GCN':
            layer_type = GCLayer
        elif GCN_type == 'GAT':
            layer_type = GATLayer
        elif GCN_type == 'HGCN':
            layer_type = HGCLayer
        elif GCN_type == 'HGAT':
            layer_type = HGATLayer
        else:
            raise AttributeError
        self.layers = torch.nn.ModuleList()
        if self.manifold is not None:
            # self.manifolds = [self.manifold] + [PoincareBall(c, learnable=False) for _ in range(depth)]
            self.manifolds = [self.manifold]*(depth+1)
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(layer_type(self.nfeat, self.nhid,self.manifolds[i],self.manifolds[i+1],edge_dim=edge_dim))
                else:
                    self.layers.append(layer_type(self.nhid, self.nhid,self.manifolds[i],self.manifolds[i+1],edge_dim=edge_dim))
        else:
            for i in range(self.depth):
                if i == 0:
                    self.layers.append(layer_type(self.nfeat, self.nhid,edge_dim=edge_dim))
                else:
                    self.layers.append(layer_type(self.nhid, self.nhid,edge_dim=edge_dim))

        self.fdim = self.nfeat + self.depth * self.nhid
        # self.fdim = self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)
        self.temb_net = MLP(num_layers=3, input_dim=self.nfeat, hidden_dim=2*self.nfeat, output_dim=self.nfeat,
                            use_bn=False, activate_func=F.elu)
        self.time_scale = nn.Sequential(
            nn.Linear(self.nfeat+1, self.nfeat),
            nn.ReLU(),
            nn.Linear(self.nfeat, 1)
        )


    def forward(self, x, adj, flags,t):
        xt = x.clone()
        temb = get_timestep_embedding(t, self.nfeat)
        x = exp_after_transp0(x,self.temb_net(temb),self.manifolds[0])
        if self.manifold is not None:
            x_list = [self.manifolds[0].logmap0(x)]
        else:
            x_list = [x]

        for i in range(self.depth):
            x = self.layers[i]((x, adj))[0]
            if self.manifold is not None:
                x_list.append(self.manifolds[i+1].logmap0(x))
            else:
                x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = self.manifold.expmap0(x)
        x = self.manifold.logmap(xt, x)
        x = x * self.time_scale(torch.cat([temb.repeat(1,x.size(1),1),self.manifold.lambda_x(xt,keepdim=True)],dim=-1))   # VE 用？

        x = mask_x(x, flags)
        return x
class ScoreNetworkX_GMH(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super().__init__()

        self.depth = depth
        self.c_init = c_init

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init, 
                                                  c_hid, num_heads, conv))
            elif _ == self.depth - 1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_final, num_heads, conv))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_hid, num_heads, conv))

        fdim = max_feat_num + depth * nhid
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=max_feat_num, 
                         use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)

        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)

        return x
