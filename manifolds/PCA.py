"""Hyperbolic dimensionality reduction models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import manifolds.euclidean as euclidean

from geoopt.manifolds.stereographic import PoincareBall as poincare


class PCA(ABC, nn.Module):
    """Dimensionality reduction model class."""

    def __init__(self, dim, n_components, lr=1e-3, max_steps=100, keep_orthogonal=False):
        super(PCA, self).__init__()
        self.dim = dim
        self.n_components = n_components
        self.components = nn.ParameterList(nn.Parameter(torch.randn(1, dim)) for _ in range(self.n_components))
        self.max_steps = max_steps
        self.lr = lr
        self.keep_orthogonal = keep_orthogonal

    def project(self, x):
        """Projects points onto the principal components."""
        Q = self.get_components()
        return self._project(x, Q)

    @abstractmethod
    def _project(self, x, Q):
        """Projects points onto the submanifold that goes through the origin and is spanned by different components.

        Args:
            x: torch.tensor of shape (batch_size, dim)
            Q: torch.tensor of shape (n_components, dim)

        Returns:
            x_p: torch.tensor of shape (batch_size, dim)
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, x, Q):
        """Computes objective to minimize.

        Args:
            x: torch.tensor of shape (batch_size, dim), data before _projection
            Q: torch.tensor of shape (n_components, dim)

        Args:
            loss: torch.tensor of shape (1,)
        """
        raise NotImplementedError

    def gram_schmidt(self, ):
        """Applies Gram-Schmidt to the component vectors."""

        def inner(u, v):
            return torch.sum(u * v)

        Q = []
        for k in range(self.n_components):
            v_k = self.components[k][0]
            proj = 0.0
            for v_j in Q:
                v_j = v_j[0]
                coeff = inner(v_j, v_k) / inner(v_j, v_j).clamp_min(1e-15)
                proj += coeff * v_j
            v_k = v_k - proj
            v_k = v_k / torch.norm(v_k).clamp_min(1e-15)
            Q.append(torch.unsqueeze(v_k, 0))
        return torch.cat(Q, dim=0)

    def orthogonalize(self):
        Q = torch.cat([self.components[i] for i in range(self.n_components)])  # (k, d)
        # _, _, v = torch.svd(Q, some=False)  # Q = USV^T
        # Q_ = v[:, :self.n_components]
        # return Q_.transpose(-1, -2)# (k, d) rows are orthonormal basis for rows of Q
        return euclidean.orthonormal(Q)

    def normalize(self, ):
        """Makes the component vectors unit-norm (not orthogonal)."""
        Q = torch.cat([self.components[i] for i in range(self.n_components)])
        return Q / torch.norm(Q, dim=1, keepdim=True).clamp_min(1e-15)

    def get_components(self, ):
        if self.keep_orthogonal:
            Q = self.gram_schmidt()
            # Q = self.orthogonalize()
        else:
            Q = self.normalize()
        return Q  # shape (n_components, dim)

    def map_to_ball(self, x):
        """Returns coordinates of _projected points in a lower-dimensional Poincare ball model.
        Args:
            x: torch.tensor of shape (batch_size, dim)

        Returns:
            torch.tensor of shape (batch_size, n_components)
        """
        Q = self.get_components()
        x_p = self._project(x, Q)
        # Q_orthogonal = self.gram_schmidt()
        Q_orthogonal = self.orthogonalize()
        return x_p @ Q_orthogonal.transpose(0, 1)

    def fit_optim(self, x, iterative=False):
        """Finds component using gradient-descent-based optimization.

        Args:
            x: torch.tensor of size (batch_size x dim)
            iterative: boolean

        Note:
            If iterative = True returns optimizes components by components (nested subspace assumption).
        """
        loss_vals = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if not iterative:
            for i in range(self.max_steps):
                # Forward pass: compute _projected variance
                Q = self.get_components()
                loss = self.compute_loss(x, Q)
                loss_vals.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1e5)
                # if self.components[0].grad.sum().isnan().item():
                optimizer.step()
        else:
            for k in range(self.n_components):
                for i in range(self.max_steps):
                    # Forward pass: compute _projected variance
                    Q = self.get_components()
                    # Project on first k components
                    loss = self.compute_loss(x, Q[:k + 1, :])
                    loss_vals.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                self.components[k].data = self.get_components()[k].unsqueeze(0)
                self.components[k].requires_grad = False
        return loss_vals

    def fit_spectral(self, x):
        """Finds component using spectral decomposition (closed-form solution).

        Args:
            x: torch.tensor of size (batch_size x dim)
        """
        raise NotImplementedError

    def fit(self, x, iterative=False, optim=True):
        """Finds principal components using optimization or spectral decomposition approaches.

        Args:
            x: torch.tensor of size (batch_size x dim)
            iterative: boolean (true to do iterative optimization of nested subspaces)
            optim: boolean (true to find components via optimization, defaults to SVD otherwise)
        """
        if optim:
            self.fit_optim(x, iterative)
        else:
            self.fit_spectral(x)



class TangentPCA(PCA):
    """Euclidean PCA in the tangent space of the mean (assumes data has Frechet mean zero)."""

    def __init__(self,manifold, dim, n_components, lr=1e-3, max_steps=100):
        super(TangentPCA, self).__init__(dim, n_components, lr, max_steps, keep_orthogonal=True)
        self.poincare = manifold

    def _project(self, x, Q):
        x_t = self.poincare.logmap0(x)  # shape (batch_size, dim)
        x_pt = (x_t @ Q.transpose(0, 1)) @ Q  # shape (batch_size, dim)
        x_p = self.poincare.expmap0(x_pt)  # shape (batch_size, dim)
        return x_p

    def compute_loss(self, x, Q):
        x_t = self.poincare.logmap0(x)  # shape (batch_size, dim)
        vals = x_t @ Q.transpose(0, 1)  # shape (batch_size, n_components)
        return - torch.sum(vals ** 2)

    def fit_spectral(self, x):
        """Geodesic PCA closed-form with SVD."""
        u = self.poincare.logmap0(x)
        S = (u.T @ u)
        U, S, V = torch.svd(S)
        for k in range(self.n_components):
            self.components[k].data = U[k:k + 1]



