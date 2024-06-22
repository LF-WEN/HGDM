import torch


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, manifold=None,node_mask=None):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        if manifold is not None:
            self.mean = manifold.expmap0(self.mean)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.node_mask = node_mask
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.manifold = manifold

    def proj_tan0(self, u):
        if self.manifold is not None and self.manifold.name == 'Lorentz':
            narrowed = u.narrow(-1, 0, 1)
            vals = torch.zeros_like(u)
            vals[:, 0:1] = narrowed
            return u - vals
        else:
            return u

    def sample(self):
        if self.manifold is None:
            x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
            x = x * self.node_mask
        else:
            mean = self.mean
            std = self.std * torch.randn(mean.shape).to(device=self.mean.device)
            std_t = self.manifold.transp0(mean,self.proj_tan0(std))
            x = self.manifold.expmap(mean,std_t)

            # std = self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
            # std = self.manifold.expmap0(self.proj_tan0(std))
            # mean = self.manifold.transp0(std, self.proj_tan0(self.mean))
            # x = self.manifold.expmap(std, mean)

            # x = self.manifold.logmap0(x)
        return x

    def kl(self):
        if self.manifold is None:
            mean = self.mean
        else:
            mean = self.manifold.logmap0(self.mean)
        if self.manifold is not None and self.manifold.name == 'Lorentz':
            kl = 0.5 * torch.mean(torch.pow(mean[...,1:], 2)
                               + self.var[...,1:] - 1.0 - self.logvar[...,1:],
                               dim=-1,keepdim=True)
        else:
            kl = 0.5 * torch.mean(torch.pow(mean, 2)
                                 + self.var - 1.0 - self.logvar,
                                 dim=-1, keepdim=True)

        kl = kl * self.node_mask

        return kl.squeeze()

    def mode(self):
        if len(self.node_mask.size()) < len(self.mean.size()):
            return self.mean * self.node_mask.unsqueeze(-1)
        else:
            return self.mean * self.node_mask


#
