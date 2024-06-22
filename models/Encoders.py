import torch
from geoopt import ManifoldParameter
from torch import nn
from models.Distributions import DiagonalGaussianDistribution
from layers.hyp_layers import get_dim_act_curv, HGCLayer, HGATLayer
from layers.layers import get_dim_act, GCLayer, GATLayer


def coord2diff(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    return torch.sqrt(radial + 1e-8)


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        self.embedding = nn.Linear(config.data.max_feat_num, config.model.hidden_dim,bias=False)
        self.mean_logvar_net = nn.Linear(config.model.hidden_dim, 2*config.model.dim)
        # self.reset_parameters()
    # def reset_parameters(self):
    #     init.xavier_uniform_(self.embedding.weight, gain=1.)
    #     init.constant_(self.embedding.bias, 0)

    def forward(self, x, adj,node_mask):
        x = self.embedding(x)  # (b,n_atom,n_atom_embed)
        output = self.encode(x,adj)
        mean_logvar = self.mean_logvar_net(output)
        posterior = DiagonalGaussianDistribution(mean_logvar, self.manifold, node_mask)
        # if self.manifold is not None:
        #     mean = self.manifold.expmap0(mean)
        #     posterior = WrappedNormal(mean,scale, self.manifold)
        # else:
        #     posterior = Normal(mean, scale)
        return posterior

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, config):
        super(GCN, self).__init__(config)
        self.dims, self.acts = get_dim_act(config, config.model.enc_layers)
        self.manifold = None
        gc_layers = []
        if config.model.layer_type == 'GCN':
            layer_type = GCLayer
        elif config.model.layer_type == 'GAT':
            layer_type = GATLayer
        else:
            raise AttributeError
        for i in range(config.model.enc_layers):
            in_dim, out_dim = self.dims[i], self.dims[i + 1]
            act = self.acts[i]
            gc_layers.append(
                layer_type(
                    in_dim, out_dim, config.model.dropout, act, config.model.edge_dim, config.model.normalization_factor,
                    config.model.aggregation_method,config.model.aggregation_method, config.model.msg_transform
                )
            )
        self.layers = nn.Sequential(*gc_layers)

    def encode(self, x,adj):
        output, _ = self.layers((x, adj))
        return output


class HGCN(Encoder):
    """
    Hyperbolic Graph Convolutional Auto-Encoders.
    """

    def __init__(self, config):
        super(HGCN, self).__init__(config)
        self.dims, self.acts, self.manifolds = get_dim_act_curv(config, config.model.enc_layers)
        self.manifold = self.manifolds[-1]
        hgc_layers = []
        if config.model.layer_type == 'HGCN':
            layer_type = HGCLayer
        elif config.model.layer_type == 'HGAT':
            layer_type = HGATLayer
        else:
            raise AttributeError
        for i in range(config.model.enc_layers):
            m_in, m_out = self.manifolds[i], self.manifolds[i + 1]
            in_dim, out_dim = self.dims[i], self.dims[i + 1]
            act = self.acts[i]
            hgc_layers.append(
                layer_type(
                    in_dim, out_dim, m_in, m_out, config.model.dropout, act, config.model.edge_dim,
                    config.model.normalization_factor,config.model.aggregation_method, config.model.msg_transform,
                    config.model.sum_transform, config.model.use_norm
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        std = 1 / torch.sqrt(torch.abs(self.manifold.k))
        self.embedding.weight = ManifoldParameter(
            self.manifolds[0].random_normal((config.data.max_feat_num, config.model.hidden_dim), std=std).T, self.manifolds[0]
        )

    def encode(self, x,adj):
        # x = self.proj_tan0(x,self.manifold)
        # x = self.manifolds[0].expmap0(x)
        output,_ = self.layers((x,adj))

        output = self.manifolds[-1].logmap0(output)

        return output

    def proj_tan0(self,u, manifold):
        if manifold.name == 'Lorentz':
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u
