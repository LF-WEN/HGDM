import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from layers.hyp_layers import HGCLayer, HGATLayer
from models.layers import DenseGCNConv, MLP
from utils.graph_utils import mask_adjs, mask_x


# -------- Graph Multi-Head Attention (GMH) --------
# -------- From Baek et al. (2021) --------
class Attention(torch.nn.Module):

    def __init__(self, in_dim, attn_dim, out_dim, num_heads=4, conv='GCN'):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv

        self.gnn_q, self.gnn_k, self.gnn_v = self.get_gnn(in_dim, attn_dim, out_dim, conv)
        self.activation = torch.tanh 
        self.softmax_dim = 2

    def forward(self, x, adj, flags, attention_mask=None):

        if self.conv == 'GCN':
            Q = self.gnn_q(x, adj)  # (B,N,attn_dim)
            K = self.gnn_k(x, adj)  # (B,N,attn_dim)
        else:
            Q = self.gnn_q(x) 
            K = self.gnn_k(x)

        V = self.gnn_v(x, adj) 
        dim_split = self.attn_dim // self.num_heads # attn_dim//head
        Q_ = torch.cat(Q.split(dim_split, 2), 0)  # ((B x num_heads),N,attn_dim//head)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.out_dim)
            A = self.activation(attention_mask + attention_score)
        else:
            A = self.activation(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.out_dim)) # (B x num_heads) x N x N
        
        # -------- (B x num_heads) x N x N --------
        A = A.view(-1, *adj.shape)
        A = A.mean(dim=0)   # ->(B,N,N)
        A = (A + A.transpose(-1,-2))/2  # 对称

        return V, A 

    def get_gnn(self, in_dim, attn_dim, out_dim, conv='GCN'):

        if conv == 'GCN':
            gnn_q = DenseGCNConv(in_dim, attn_dim)
            gnn_k = DenseGCNConv(in_dim, attn_dim)
            gnn_v = DenseGCNConv(in_dim, out_dim)

            return gnn_q, gnn_k, gnn_v

        elif conv == 'MLP':
            num_layers=2
            gnn_q = MLP(num_layers, in_dim, 2*attn_dim, attn_dim, activate_func=torch.tanh)
            gnn_k = MLP(num_layers, in_dim, 2*attn_dim, attn_dim, activate_func=torch.tanh)
            gnn_v = DenseGCNConv(in_dim, out_dim)

            return gnn_q, gnn_k, gnn_v

        else:
            raise NotImplementedError(f'{conv} not implemented.')


class HAttention(torch.nn.Module):

    def __init__(self, in_dim, attn_dim, out_dim, num_heads=4, conv='GCN',manifold=None):
        super(HAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv
        self.manifold = manifold
        self.gnn_q, self.gnn_k, self.gnn_v = self.get_gnn(in_dim, attn_dim, out_dim, conv,manifold)
        self.activation = torch.tanh
        self.softmax_dim = 2
        self.r = nn.Parameter(torch.ones((num_heads,))*5)
        self.t = nn.Parameter(torch.ones((num_heads,)))

    def forward(self, x, adj, flags):

        Q = self.gnn_q((x, adj))  # (b,n,n_head,attn_dim/n_head)
        # K = self.gnn_k((x, adj))
        V,_ = self.gnn_v((x, adj))
        dist = self.manifold.dist(Q[:,:,None,:,:],Q[:,None,:,:,:]).view(-1, *adj.shape)  # (n_head,b,n,n)
        A = self.activation((self.r[:,None,None,None]-dist)*self.t[:,None,None,None]) # (n_head,b,n,n)

        # -------- (B x num_heads) x N x N --------
        A = A.view(-1, *adj.shape)
        A = A.mean(dim=0)
        A = (A + A.transpose(-1, -2)) / 2

        return V, A

    def get_gnn(self, in_dim, attn_dim, out_dim, conv='GCN',manifold=None):

        if conv == 'GCN':
            gnn_q = HGATLayer(in_dim, attn_dim,manifold,manifold,num_of_heads=self.num_heads,use_act=False,return_multihead=True)
            gnn_k = HGATLayer(in_dim, attn_dim,manifold,manifold,num_of_heads=self.num_heads,use_act=False,return_multihead=True)
            gnn_v = HGATLayer(in_dim, out_dim,manifold,manifold,num_of_heads=self.num_heads,use_act=False)

            return gnn_q, gnn_k, gnn_v

        elif conv == 'MLP':
            raise NotImplementedError(f'{conv} not implemented.')
            # num_layers = 2
            # gnn_q = MLP(num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh)
            # gnn_k = MLP(num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh)
            # gnn_v = DenseGCNConv(in_dim, out_dim)
            #
            # return gnn_q, gnn_k, gnn_v

        else:
            raise NotImplementedError(f'{conv} not implemented.')

# -------- Layer of ScoreNetworkA --------
class AttentionLayer(torch.nn.Module):

    def __init__(self, num_linears, conv_input_dim, attn_dim, conv_output_dim, input_dim, output_dim, 
                    num_heads=4, conv='GCN'):

        super(AttentionLayer, self).__init__()

        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn_dim = attn_dim
            self.attn.append(Attention(conv_input_dim, self.attn_dim, conv_output_dim,
                                        num_heads=num_heads, conv=conv))

        self.hidden_dim = 2*max(input_dim, output_dim)
        self.mlp = MLP(num_linears, 2*input_dim, self.hidden_dim, output_dim, use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, 
                                    use_bn=False, activate_func=F.elu)

    def forward(self, x, adj, flags):
        """

        :param x:  B x N x F_i
        :param adj: B x C_i x N x N
        :return: x_out: B x N x F_o, adj_out: B x C_o x N x N
        """
        mask_list = []
        x_list = []
        for _ in range(len(self.attn)):
            _x, mask = self.attn[_](x, adj[:,_,:,:], flags)
            mask_list.append(mask.unsqueeze(-1))
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0,2,3,1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
        _adj = _adj + _adj.transpose(-1,-2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out

class HAttentionLayer(torch.nn.Module):
    # 改成用双曲特征更新邻接矩阵
    def __init__(self, num_linears, conv_input_dim, attn_dim, conv_output_dim, input_dim, output_dim,
                    num_heads=4, conv='GCN',manifold=None):
        """
        :param num_linears:
        :param conv_input_dim: 输入的节点dim
        :param attn_dim:
        :param conv_output_dim:
        :param input_dim: 邻接矩阵要cat的dim
        :param output_dim:
        :param num_heads:
        :param conv:
        :param manifold:
        """

        super(HAttentionLayer, self).__init__()
        self.manifold = manifold
        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn_dim = attn_dim
            self.attn.append(HAttention(conv_input_dim, self.attn_dim, conv_output_dim,
                                        num_heads=num_heads, conv=conv,manifold=manifold))

        self.hidden_dim = 2*max(input_dim, output_dim)
        self.mlp = MLP(num_linears, 2*input_dim, self.hidden_dim, output_dim, use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim,
                                    use_bn=False, activate_func=F.elu)

    def forward(self, x, adj, flags):
        """

        :param x:  B x N x F_i
        :param adj: B x C_i x N x N
        :return: x_out: B x N x F_o, adj_out: B x C_o x N x N
        """
        mask_list = []
        x_list = []
        for _ in range(len(self.attn)):
            _x, mask = self.attn[_](x, adj[:,_,:,:], flags)
            mask_list.append(mask.unsqueeze(-1))
            x_list.append(self.manifold.logmap0(_x))
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = self.manifold.expmap0(x_out)
        # x_out = torch.tanh(x_out)

        mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0,2,3,1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
        _adj = _adj + _adj.transpose(-1,-2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out