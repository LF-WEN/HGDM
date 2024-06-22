import torch

from manifolds.utils import exp_after_transp0
from models.ScoreNetwork_X import ScoreNetworkX_poincare
from sde_graph_lib import VPSDE, VESDE, subVPSDE
from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise


def get_score_fn(sde, model, train=True, continuous=True):

    if not train:
        model.eval()
    model_fn = model

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        def score_fn(x, adj, flags, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous:
                labels = t * 999
                score = model_fn(x, adj, flags,labels)
                std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
            else:
                raise NotImplementedError(f"Discrete not supported")
            # if not isinstance(model_fn, ScoreNetworkX_poincare):    # Todo
            score = -score / std
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, adj, flags, t):
            if continuous:
                labels = sde.T - t
                labels *= sde.N - 1
                score = model_fn(x, adj, flags,labels)
            else:
                raise NotImplementedError(f"Discrete not supported")

            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

    return score_fn


def get_sde_loss_fn(sde_x, sde_adj, train=True, reduce_mean=False, continuous=True,
                    likelihood_weighting=False, eps=1e-5,manifold=None,encoder=None):

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model_x, model_adj, x, adj):
        flags = node_flags(adj)
        if encoder is not None:
            posterior = encoder(x,adj,flags)
            x = posterior.mode()
        else:
            if manifold is not None:
                x = manifold.expmap0(x)
        # print('radius:',manifold.radius)
        # print('x:',x[0])
        # print('mean:',x.mean())
        # print('max:',x.max())
        # exit(0)
        score_fn_x = get_score_fn(sde_x, model_x, train=train, continuous=continuous)
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=train, continuous=continuous)

        t = torch.rand((adj.shape[0],1,1), device=adj.device) * (sde_adj.T - eps) + eps

        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = sde_x.marginal_prob(x, t)

        if manifold is not None:
            perturbed_x = exp_after_transp0(mean_x,std_x * z_x,manifold)
        else:
            perturbed_x = mean_x + std_x * z_x
        perturbed_x = mask_x(perturbed_x, flags)

        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)

        score_x = score_fn_x(perturbed_x, perturbed_adj, flags, t)
        score_adj = score_fn_adj(perturbed_x, perturbed_adj, flags, t)

        if not likelihood_weighting:
            if manifold is not None:
                with torch.enable_grad():
                    xt = perturbed_x.detach()
                    xt.requires_grad = True
                    u = manifold.logmap(mean_x, xt)
                    v = manifold.transp0back(mean_x, u)
                    dim = v.size(-1)
                    dist = manifold.dist(mean_x, xt, keepdim=True)
                    sqrt_c_dist = dist * torch.sqrt(torch.abs(manifold.k)) + 1e-6   #add eps avoiding nan
                    logp = -1 * v ** 2 / (2 * std_x ** 2).sum(-1,keepdims=True) + (dim - 1) * torch.log(sqrt_c_dist / torch.sinh(sqrt_c_dist))
                    target, = torch.autograd.grad(logp.sum(), xt)
                    target = mask_x(target, flags)
                losses_x = torch.square(score_x-target)
            else:
                losses_x = torch.square(score_x * std_x + z_x)  #std(score+z/std)
            losses_x = reduce_op(losses_x, dim=-1)

            losses_adj = torch.square(score_adj * std_adj + z_adj)
            # losses_adj = reduce_op(losses_adj.reshape(losses_x.shape[0], -1), dim=-1)
        else:
            g2_x = sde_x.sde(torch.zeros_like(x), t)[1] ** 2
            losses_x = torch.square(score_x + z_x / std_x[:, None, None])
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1) * g2_x

            g2_adj = sde_adj.sde(torch.zeros_like(adj), t)[1] ** 2
            losses_adj = torch.square(score_adj + z_adj / std_adj[:, None, None])
            losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1) * g2_adj

        adj_flags = flags.unsqueeze(2) * flags.unsqueeze(1)  # b*n
        return torch.sum(losses_x.view(-1)) / flags.sum(), torch.sum(losses_adj.view(-1)) / adj_flags.sum()
        # return torch.mean(losses_x), torch.mean(losses_adj)

    return loss_fn