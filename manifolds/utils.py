import torch


def exp_after_transp0(x, y, manifold, x_in_manifold=True, y_in_manifold=False):
    """
    :param x: start point,should in manifold
    :param y: to be transported,should in tangent space of zero point
    :param manifold:
    :param x_in_manifold:
    :param y_in_manifold:
    :return: x+y in manifold
    """

    if not x_in_manifold:
        x = manifold.expmap0(proj_tan0(x,manifold))
    if y_in_manifold:
        y = manifold.logmap0(y)
    y = manifold.transp0(x, proj_tan0(y,manifold))
    return manifold.expmap(x, y)


def transp0back_after_logmap(x, y, manifold, x_in_manifold=True, y_in_manifold=True):
    """
    implement more accurate y-x in manifold for wrap_dist_variant
    :param x:
    :param y:
    :param manifold:
    :param x_in_manifold:
    :param y_in_manifold:
    :return:
    """
    if not x_in_manifold:
        x = manifold.expmap0(proj_tan0(x,manifold))
    if not y_in_manifold:
        y = manifold.expmap0(proj_tan0(y,manifold))
    y = manifold.logmap(x, y)
    y = manifold.transp0back(x, y)
    return manifold.expmap0(y)

def proj_tan0(u, manifold):
    if manifold.name == 'Lorentz':
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
    else:
        return u

def proj_tan(x, u,manifold):
    if manifold.name == 'Lorentz':
        eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        d = x.size(-1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=-1, keepdim=True)   # sum(x[:,1:]*u[:,1:],dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=eps[x.dtype])
        return vals + mask * u
    else:
        return u

def mobius_scalar_mul(scalar, x, manifold):
    if manifold.name == 'Lorentz':
        x = manifold.logmap0(x)
        return manifold.expmap0(proj_tan0(scalar * x, manifold))
    else:
        return manifold.mobius_scalar_mul(scalar, x)

def mobius_sub(x,y,manifold,x_in_manifold=True, y_in_manifold=True):
    """x-y"""
    if not x_in_manifold:
        x = manifold.expmap0(proj_tan0(x,manifold))
    if not y_in_manifold:
        y = manifold.expmap0(proj_tan0(y,manifold))
    if manifold.name == 'Lorentz':
        return exp_after_transp0(x, mobius_scalar_mul(-1,y,manifold), manifold,True,True)
    else:
        return manifold.mobius_sub(x, y)

def mobius_add(x,y,manifold,x_in_manifold=True, y_in_manifold=True):
    """x+y"""
    if not x_in_manifold:
        x = manifold.expmap0(proj_tan0(x,manifold))
    if not y_in_manifold:
        y = manifold.expmap0(proj_tan0(y,manifold))
    if manifold.name == 'Lorentz':
        return exp_after_transp0(x, y, manifold,True,True)
    else:
        return manifold.mobius_add(x, y)

def lorentz_to_poincare(x, m):
    K = m.k
    sqrtK = K ** 0.5
    d = x.size(-1) - 1
    return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

def poincare_to_lorentz(x, m):
    K = 1. / m.c
    sqrtK = K ** 0.5
    sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
    return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=-1) / (K - sqnorm)

def unsqueeze_tangent(x):
    return torch.cat((torch.zeros_like(x[..., 0]).unsqueeze(-1), x), dim=-1)