"""Example dynamical systems for topological equivalence analysis."""

# %% Imports

from scipy.io import loadmat
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.autograd.functional import jvp
from typing import Any, Sequence, Callable
from DFORM import Model


# %% Dynamical systems

class WongWang(Model):
    """(Wong & Wang, 2006) decision-making model.
    
    The bifurcation parameter is cprime (0 to 1). See Figure 5 in the paper.
    The dt is 1 ms.
    """
    def __init__(self, cprime=0.):
        super().__init__(dim=2)
        assert 0 <= cprime <= 1
        self.cprime = cprime
        self.JAext = 5.2e-4  # [nA/Hz]
        self.mu0 = 30.  # [Hz]
        self.I0 = 0.3255  # [nA]
        self.I1 = self.JAext * self.mu0 * (1 + self.cprime)
        self.I2 = self.JAext * self.mu0 * (1 - self.cprime)
        self.xBias = self.I0 + Tensor([self.I1, self.I2])
        self.JN = Tensor([
            [0.2609, -0.0497],
            [-0.0497, 0.2609]
        ])  # self.x = [S1, S2] @ self.JN.T + self.xBias, [nA]
        self.a = 270.  # [1/VnC]
        self.b = 108.  # [Hz]
        self.d = 0.154  # [s]
        # Define y = ax - b = yw @ [[S1], [S2]] + yBias
        self.yw = self.a * self.JN
        self.yBias = self.a * self.xBias - self.b
        self.gamma = 0.641 / 1000  # To change H from [1/s] to [1/ms]
        self.taus = 100.  # [ms]
    def forward(self, S):
        y = S @ self.yw.T + self.yBias
        H = y / (1 - torch.exp(-self.d * y))
        dS = -S / self.taus + (1 - S) * self.gamma * H
        return dS
    def get_samples(self, n: int) -> Tensor:
        return torch.rand(n, 2)


class SaddleNode(Model):
    """Sadddle-node bifurcation at mu = 0."""
    def __init__(self, mu=0.):
        super().__init__(dim=2)
        self.mu = mu
    def forward(self, x):
        return torch.stack([
            self.mu - x[..., 0] ** 2,
            -x[..., 1]
        ], dim=-1)
    

class SuperPitch(Model):
    """Supercritical pitchfork bifurcation at mu = 0."""
    def __init__(self, mu=0.):
        super().__init__(dim=2)
        self.mu = mu
    def forward(self, x):
        return torch.stack([
            self.mu * x[..., 0] - x[..., 0] ** 3,
            -x[..., 1]
        ], dim=-1)
    def get_samples(self, n: int) -> Tensor:
        if self.mu <= 0:  # Stable
            return torch.randn(n, 2)
        else:  # Bistable at x = +-sqrt(mu)
            x0 = np.sqrt(self.mu)
            return torch.stack([torch.rand(n) * 3 * x0 - 1.5 * x0, torch.randn(n) * x0], dim=-1)


class TransPitch(Model):
    """Transcritical pitchfork bifurcation at mu = 0.
    
    mu < 0: Saddle at (mu, 0), node at the origin
    mu = 0: Degenerated node at the origin
    mu > 0: Saddle at the origin, node at (mu, 0)
    """
    def __init__(self, mu=0.):
        super().__init__(dim=2)
        self.mu = mu
    def forward(self, x):
        return torch.stack([
            self.mu * x[..., 0] - x[..., 0] ** 2,
            -x[..., 1]
        ], dim=-1)
    def get_samples(self, n: int) -> Tensor:
        if self.mu == 0:
            return torch.randn(n, 2)
        else:  # Two fixed points at (0, 0) and (mu, 0)
            x0 = self.mu
            return torch.stack([torch.rand(n) * 3 * x0 - 1 * x0, torch.randn(n) * x0], dim=-1)


class SubPitch(Model):
    """Subcritical pitchfork bifurcation at mu = 0."""
    def __init__(self, mu=0.):
        super().__init__(dim=2)
        self.mu = mu
    def forward(self, x):
        return torch.stack([
            self.mu * x[..., 0] + x[..., 0] ** 3,
            -x[..., 1]
        ], dim=-1)


class SuperHopf(Model):
    """Supercritical Hopf bifurcation at mu = 0."""
    def __init__(self, mu=0., omega=1.):
        super().__init__(dim=2)
        self.mu = mu
        self.omega = omega
    def forward(self, x):
        MuSubR2 = self.mu - torch.sum(x ** 2, dim=-1)
        return torch.stack([
            MuSubR2 * x[..., 0] - self.omega * x[..., 1],
            self.omega * x[..., 0] + MuSubR2 * x[..., 1]
        ], dim=-1)
    def get_samples(self, n: int) -> Tensor:
        if self.mu <= 0:  # Stable
            return torch.randn(n, 2)
        else:  # Limit cycle of radius sqrt(mu)
            r0 = np.sqrt(self.mu)
            r = torch.rand(n) * r0 * 0.4 + r0 * 0.8
            theta = torch.rand(n) * 2 * np.pi
            return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


class SubHopf(Model):
    """Subcritical Hopf bifurcation at mu = 0."""
    def __init__(self, mu=0.):
        super().__init__(dim=2)
        self.mu = mu
    def forward(self, x):
        return torch.stack([
            self.mu * x[..., 0] - x[..., 1] + x[..., 0] * (x[..., 1] ** 2),
            x[..., 0] + self.mu * x[..., 1] + x[..., 1] ** 3
        ], dim=-1)


class VanDerPol(Model):
    """Van der Pol oscillator. Non-generic bifurcation at mu = 0."""
    def __init__(self, mu=0.):
        super().__init__(dim=2)
        self.mu = mu
    def forward(self, x):
        return torch.stack([
            x[..., 1],
            self.mu * (1 - x[..., 0] ** 2) * x[..., 1] - x[..., 0]
        ], dim=-1)
    def get_samples(self, n: int) -> Tensor:
        if self.mu <= 0:
            return torch.randn(n, 2)
        else:  # Limit cycle going through (+-2, 0) and roughly (+-1, +-(mu+2))
            xrange = 3
            yrange = 1.5 * (self.mu + 2)
            return torch.stack([
                torch.rand(n) * 2 * xrange - xrange,
                torch.rand(n) * 2 * yrange - yrange
            ], dim=-1)
    

class Homoclinic(Model):
    """Homoclinic bifurcation at mu = -0.8645"""
    def __init__(self, mu=-0.8645):
        super().__init__(dim=2)
        self.mu = mu
    def forward(self, x):
        return torch.stack([
            x[..., 1],
            self.mu * x[..., 1] + x[..., 0] - x[..., 0] ** 2 + x[..., 0] * x[..., 1]
        ], dim=-1)


class SNIC(Model):
    """Saddle-node on invariant cycle bifurcation at mu = 1.
    
    Dynamics are given in polar coordinates by:
        dr/dt = r * (1 - r^2)
        dtheta/dt = -abs(sin(theta)) + mu
    """
    def __init__(self, mu=1.):
        super().__init__(dim=2)
        self.mu = mu
    def forward(self, x):
        allx, ally = x[..., 0], x[..., 1]
        allr = torch.sqrt(allx ** 2 + ally ** 2)
        allthetasin = ally / allr
        allthetacos = allx / allr
        allrdot = allr * (1 - allr ** 2)
        allthetadot = -abs(allthetasin) + self.mu
        allxdot = allrdot * allthetacos - allr * allthetasin * allthetadot
        allydot = allrdot * allthetasin + allr * allthetacos * allthetadot
        # Handle (0, 0)
        allxdot[torch.isnan(allxdot)] = 0.0
        allydot[torch.isnan(allydot)] = 0.0
        return torch.stack([allxdot, allydot], dim=-1)
    def get_samples(self, n: int) -> Tensor:
        r = torch.rand(n) * 0.4 + 0.8
        theta = torch.rand(n) * 2 * np.pi
        return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


class LoktaVolterra(Model):
    """Lokta-Volterra predator-prey model.
    
    The original equations are:
        dx/dt = x * (a - b * y)
        dy/dt = -y * (c - d * x)
    
    Bifurcation happens at a=0.
    """
    def __init__(self, a=2, b=1.2, c=1, d=0.9):
        super().__init__(dim=2)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    def forward(self, x):
        return torch.stack([
            x[..., 0] * (self.a - self.b * x[..., 1]),
            -x[..., 1] * (self.c - self.d * x[..., 0])
        ], dim=-1)


class Lorenz(Model):
    """Lorenz system."""
    def __init__(self, sigma=10, rho=28, beta=8/3):
        super().__init__(dim=3)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    def forward(self, x):
        return torch.stack([
            self.sigma * (x[..., 1] - x[..., 0]),
            x[..., 0] * (self.rho - x[..., 2]) - x[..., 1],
            x[..., 0] * x[..., 1] - self.beta * x[..., 2]
        ], dim=-1)


class LinearAutonomous(Model):
    """Linear autonomous system."""
    def __init__(self, A):
        super().__init__(A.shape[0])
        self.device = A.device
        assert len(A.shape) == 2 and A.shape[0] == A.shape[1], 'A must be a square matrix!'
        self.A = nn.Parameter(A, requires_grad=False)
    def forward(self, x):
        return x @ self.A.T
    def get_A(self):
        return self.A

class RNN(Model):
    """Continuous-time vanilla RNN."""
    def __init__(self, n: int, act: nn.Module = torch.tanh, W: Tensor | None = None, D: Tensor | None = None):
        super().__init__(n)
        if W is None:
            W = torch.randn(n, n)
        self.device = W.device
        if D is None:
            D = torch.randn(n, device=self.device)
        self.W = nn.Parameter(W, requires_grad=False)
        self.D = nn.Parameter(D.flatten(), requires_grad=False)
        self.act = act
        self.n = n
    def forward(self, x: Tensor) -> Tensor:
        return self.act(x) @ self.W.T - self.D * x


# %% Transformed dynamical systems

class Transformed(Model):
    """Dynamic system transformed by a DFORM model y = h(x)."""
    def __init__(self, f: Model, h: Model, inverse: bool = False):
        """Initialize a transformed dynamical system y = h(x).

        Args:
            f: the original dynamical system x = f(x).
            h: the transformation y = h(x). Must have a "deform" and a "inv_deform" method.
              (e.g., a DFORM model)
            inverse: if True, the transformation is y = h^{-1}(x) instead.
        """
        super().__init__(f.dim)
        self.device = f.device
        self.f = f
        self.h = h
        self.inverse = inverse
        if self.inverse:
            self.deform = self.h.inv_deform
            self.inv_deform = self.h.deform
        else:
            self.deform = self.h.deform
            self.inv_deform = self.h.inv_deform
    def forward(self, y: Tensor):
        x = self.inv_deform(y)
        fx = self.f(x)
        _, gy = jvp(self.deform, (x,), (fx,), create_graph=True)   # Needs to create graph to compute Jacobians etc.
        return gy
    def get_samples(self, *args, **kwargs):
        return self.deform(self.f.get_samples(*args, **kwargs))
    def to(self, dvc: str|torch.device):
        """Move the model to a different device."""
        super().to(dvc)
        self.device = dvc
        self.f.to(dvc)
        self.h.to(dvc)
        return self


class LinearTransformed(Model):
    """Linearly transformed dynamical system y = Hx + bias"""
    def __init__(self, f: Model, H: Tensor, bias: Tensor | None = None):
        super().__init__(f.dim)
        self.device = f.device
        self.f = f
        self.H = nn.Parameter(H.to(self.device), requires_grad=False)
        self.HInv = nn.Parameter(torch.linalg.inv(self.H), requires_grad=False)
        self.bias = bias.to(self.device) if bias is not None else None
    def forward(self, x: Tensor):
        if self.bias is not None:
            x = x - self.bias
        return self.f(x @ self.HInv.T) @ self.H.T
    def get_samples(self, *args, **kwargs):
        if self.bias is not None:
            return self.f.get_samples(*args, **kwargs).to(self.device) @ self.H.T + self.bias
        else:
            return self.f.get_samples(*args, **kwargs).to(self.device) @ self.H.T


class SinhTransformed(Model):
    """Sinh-transformed dynamical system."""
    def __init__(self, f: Model):
        super().__init__(f.dim)
        self.device = f.device
        self.f = f
    def forward(self, x: Tensor):
        x = torch.asinh(x)
        return self.f(x) * torch.cosh(x)
    def get_samples(self, *args, **kwargs):
        return torch.sinh(self.f.get_samples(*args, **kwargs))


class ExpDamping(Model):
    """Exponential damping to a dynamical system."""
    def __init__(self, f: Model, tau: float = 1):
        super().__init__(f.dim)
        self.device = f.device
        self.f = f
        self.tau = tau
    def forward(self, x: Tensor):
        return self.f(x) * torch.exp(
            -torch.norm(x, dim=-1, keepdim=True) / self.tau)


class Noisy(Model):
    """Additive noise to a dynamical system."""
    def __init__(self, f: Model, sigma: float = 0.1):
        super().__init__(f.dim)
        self.device = f.device
        self.f = f
        self.sigma = sigma
    def forward(self, x: Tensor):
        return self.f(x) + self.sigma * torch.randn_like(x)


class LinearProject(Model):
    """Full-rank linear projection of a dynamical system.

    If the old system is dx/dt = f(x), and the new coordinates are y = Ax,
    where A is a full-rank matrix of size (nNew, nOld), then dy/dt = Af(xHat),
    where xHat = A^{+}y is the minimum-norm least squares solution to y = Ax.

    1. If nNew < nOld, there are infinite many xhat that satisfy y = A * xHat.
    For example, if A is the first nNew PCs, then xHat will be the linear
    combination of first nNew PCs with y as the coefficients. Therefore, the
    dynamics of y are actually the dynamics of x restricted to the subspace
    spanned by the columns of A.

    2. If nNew > nOld, xHat will be the projection of y onto the subspace
    spanned by the columns of A. Therefore, the dynamics of y are the dynamics
    of x when projected onto this subspace, and the dynamics of y projected
    onto the orthogonal complement of this subspace is always zero. Namely, the
    dynamics of y are restricted to the subspace spanned by the columns of A.
    """
    def __init__(self, f: nn.Module, A: Tensor, bias: Tensor | None = None):
        super().__init__(A.shape[0])
        self.device = f.device
        self.f = f
        self.A = nn.Parameter(A.to(self.device), requires_grad=False)
        self.bias = bias.to(self.device) if bias is not None else None
        self.nNew, self.nOld = A.shape
        # Check if the projection is full-rank
        assert torch.linalg.matrix_rank(A) == min(self.nNew, self.nOld), 'A must be full-rank!'
    def forward(self, x: Tensor):
        return self.f(torch.linalg.solve(self.A.T, x.T).T) @ self.A.T


class Composed(Model):
    """Composition of several dynamical systems.

    Slice the input x into several parts, and pass each part to a different
    dynamical system. Concatenate the outputs of these systems as the output.
    """
    def __init__(self, fs: Sequence[Model]):
        all_dims = [f.dim for f in fs]
        super().__init__(dim=sum(all_dims))
        self.device = fs[0].device
        self.fs = nn.ModuleList(fs)
        self.all_dims = all_dims
        self.dim_start = [sum(all_dims[:i]) for i in range(len(all_dims))]
        self.dim_end = [sum(all_dims[:i+1]) for i in range(len(all_dims))]
    def forward(self, x: Tensor):
        return torch.cat([f(x[:, s:e]) for f, s, e in
                          zip(self.fs, self.dim_start, self.dim_end)], dim=-1)
    def get_samples(self, n: int):
        return torch.cat([f.get_samples(n) for f in self.fs], dim=-1)


class Restricted(Model):
    """High-dimensional model restricted to a subset of the dimensions"""
    def __init__(self, model: Model, idx: Sequence[int]):
        super().__init__(dim=len(idx))
        self.device = model.device
        self.model = model
        self.full_dim = model.dim
        self.idx = idx
    def forward(self, x: Tensor) -> Tensor:
        x_extended = torch.zeros(*x.shape[:-1], self.full_dim, device=x.device)
        x_extended[..., self.idx] = x
        return self.model(x_extended)[..., self.idx]
    def get_samples(self, n: int) -> Tensor:
        return self.model.get_samples(n)[..., self.idx]


# %% Variants of RNNs

class cGRU(Model):
    """GRU as a continuous-time dynamical system.

    See https://www.frontiersin.org/articles/10.3389/fncom.2021.678158
    for details. Here we eliminate the terms that only influence
    the magnitude (but not direction) of speed, i.e., the z term.
    """
    def __init__(self, Uh, Ur, bh, br):
        super().__init__(dim=Uh.shape[0])
        self.device = Uh.device
        self.Uh = nn.Parameter(Uh, requires_grad=False)
        self.Ur = nn.Parameter(Ur, requires_grad=False)
        self.bh = nn.Parameter(bh, requires_grad=False)
        self.br = nn.Parameter(br, requires_grad=False)
    def forward(self, h):
        r = torch.sigmoid(h @ self.Ur.T + self.br)
        return torch.tanh((r * h) @ self.Uh.T + self.bh) - h


class MINDy(Model):
    """Mesoscale Individualized NeuroDynamic model.

    See https://doi.org/10.1016/j.neuroimage.2020.117046 for details.
    """
    def __init__(self, W: Tensor, alpha: Tensor, D: Tensor, b: float | Tensor = 20/3):
        super().__init__(dim=W.shape[0])
        self.device = W.device
        self.W = nn.Parameter(W, requires_grad=False)
        self.alpha = nn.Parameter(alpha.flatten(), requires_grad=False)
        self.D = nn.Parameter(D.flatten(), requires_grad=False)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=False)
    def forward(self, x):
        psix = (torch.sqrt(self.alpha ** 2 + (self.b * x + 0.5) ** 2) -
                torch.sqrt(self.alpha ** 2 + (self.b * x - 0.5) ** 2))
        return psix @ self.W.T - self.D * x


# %% Functions to generate test linear systems

def eig2real(eig_real: Tensor, eig_imag: Tensor | None = None) -> Tensor:
    """Convert complex conjugate eigenvalues to a real matrix.
    
    For simplicity, we assume the complex eigenvalues are in position 0, 2, 4, ...
    and their conjugates in position 1, 3, 5, ...
    """
    if eig_imag is None:
        return torch.diag(eig_real)
    
    assert eig_real.shape == eig_imag.shape, 'Real and imaginary parts must have the same shape!'
    nComplex = len(torch.where(eig_imag != 0)[0])
    assert (
        nComplex % 2 == 0 and
        torch.allclose(eig_imag[:nComplex:2], -eig_imag[1:nComplex:2]) and
        torch.allclose(eig_real[:nComplex:2], eig_real[1:nComplex:2])
    ), 'Complex eigenvalues must be conjugate pairs and listed first!'

    A = torch.diag(eig_real)
    for i in range(0, nComplex, 2):
        A[i, i + 1] = eig_imag[i]
        A[i + 1, i] = eig_imag[i + 1]
    return A

def get_good_cond_matrix(n: int, ortho: bool = False, orient: bool = False,
                         volume: bool = False, dvc: str|torch.device = 'cpu') -> Tensor:
    """Generate a `n`-dimensional random matrix with good condition number.
    
    Additionally:
        - `ortho`: whether to make it orthogonal.
        - `orient`: whether to make it orientation-preserving (i.e., det > 0).
        - `volume`: whether to make it volume-preserving (i.e., |det| = 1).
        - `dvc`: device to put the matrix on.
    """
    thres = n * 3
    for _ in range(100):
        A = torch.randn(n, n, device=dvc) / np.sqrt(n)
        if torch.linalg.cond(A) < thres:
            if ortho:
                A = torch.linalg.qr(A)[0]
            sgn_detA, logdetA = torch.linalg.slogdet(A)
            root_detA = torch.exp(logdetA / n)
            if orient:
                A[-1, :] *= sgn_detA
            if volume and not ortho:
                A /= root_detA
            return A
    raise RuntimeError('Cannot find a matrix with good condition number!')

def get_matrix_sgnvec(A: Tensor) -> tuple[Sequence[int], Sequence[int]]:
    """Get the sign vectors of the eigenvalues of a matrix."""
    eps = 1e-6
    eigs = torch.linalg.eigvals(A)
    is_real = torch.abs(eigs.imag) < eps
    is_complex = torch.abs(eigs.imag) >= eps
    is_positive = eigs.real > eps
    is_negative = eigs.real < -eps
    is_zero = torch.abs(eigs.real) <= eps
    sgn_real = [
        torch.sum(torch.logical_and(is_real, is_positive)).item(),
        torch.sum(torch.logical_and(is_real, is_negative)).item(),
        torch.sum(torch.logical_and(is_real, is_zero)).item()
    ]
    sgn_complex = [
        torch.sum(torch.logical_and(is_complex, is_positive)).item(),
        torch.sum(torch.logical_and(is_complex, is_negative)).item(),
        torch.sum(torch.logical_and(is_complex, is_zero)).item()
    ]
    return sgn_real, sgn_complex

def get_sgnvec_matrix(n: int = 0, sgn_real: Sequence[int] = [0, 0, 0],
        sgn_complex: Sequence[int] = [0, 0, 0], dvc: str|torch.device = 'cpu'
        ) -> Tensor:
    """Generate a square matrix with specified eigenvalue signs.

    Args:
        n: Dimension of the state space.
        sgn_real: Three-element vector, number of **REAL** eigenvalues
            being positive, negative, and zeros, respectively.
        sgn_complex: Three-element vector, number of **COMPLEX** eigenvalues
            having positive, negative, and zero **real parts**, respectively.
            All should be even numbers.
        dvc: Device to put the matrix on.
    
    Returns:
        A1: A matrix with the specified eigenvalue signs.
    
    Examples:
        - n*n matrix without requirements on the eigenvalues:
            A1 = get_similar_LTIs(n)
        - 6*6 matrix with one eigenvalue being real positive,
        one negative real, two being zeros, and two on left half plane:
            A1 = get_similar_LTIs(6, [1, 1, 2], [0, 2, 0])
    """
    if isinstance(sgn_real, torch.Tensor):
        sgn_real = [int(curr) for curr in sgn_real.tolist()]
    if isinstance(sgn_complex, torch.Tensor):
        sgn_complex = [int(curr) for curr in sgn_complex.tolist()]
    assert len(sgn_real) == 3 and len(sgn_complex) == 3, 'Must provide 3-element sign vectors!'
    n_real, n_complex = sum(sgn_real), sum(sgn_complex)
    if n_real + n_complex == 0:
        if n == 0:
            raise ValueError('Must provide either n or sign vectors!')
        else:
            A1 = get_good_cond_matrix(n)
    else:
        if n != 0:
            assert n == n_real + n_complex
        else:
            n = n_real + n_complex
        # Real eigenvalues
        eig_real = torch.cat([
            torch.rand(sgn_real[0]),
            -torch.rand(sgn_real[1]),
            torch.zeros(sgn_real[2])
        ])
        # Complex eigenvalues
        eig_complex_real = torch.cat([
            torch.rand(sgn_complex[0]),
            -torch.rand(sgn_complex[1]),
            torch.zeros(sgn_complex[2])
        ])
        eig_complex_imag = torch.rand(n_complex)
        # Enforce conjugacy
        eig_complex_real[1::2] = eig_complex_real[::2]
        eig_complex_imag[1::2] = -eig_complex_imag[::2]
        # Combine
        eig_all_real = torch.cat([eig_complex_real, eig_real])
        eig_all_imag = torch.cat([eig_complex_imag, torch.zeros_like(eig_real)])
        A1 = eig2real(eig_all_real, eig_all_imag)

    # Do an orthogonal transformation on A1
    H = get_good_cond_matrix(n, ortho=True)
    A1 = H @ torch.linalg.solve(H.T, A1.T).T
    A1 = A1.to(dvc)
    return A1

def get_similar_LTIs(n: int = 0, sgn_real: Sequence[int] = [0, 0, 0],
        sgn_complex: Sequence[int] = [0, 0, 0], dvc: str|torch.device = 'cpu',
        **kwargs) -> tuple[LinearAutonomous, LinearAutonomous, Tensor]:
    """Generate two similar linear time-invariant systems.

    Args:
        n, sgn_real, sgn_complex, dvc: see `get_sgnvec_matrix`.
        **kwargs: passed to `get_good_cond_matrix` for the transformation
            (ortho, orient, volume)
    
    Returns:
        A1, A2: Two similar linear autonomous systems.
        H: The similarity transformation matrix. A2 = H @ A1 @ inv(H).
    """
    A1 = get_sgnvec_matrix(n, sgn_real, sgn_complex, dvc=dvc)
    H = get_good_cond_matrix(n, dvc=dvc, **kwargs)
    A2 = H @ torch.linalg.solve(H.T, A1.T).T
    return LinearAutonomous(A1), LinearAutonomous(A2), H


# %% Functions to generate test RNNs

def get_RNN(N: int = 16, K: int = 1, g: float = 0.9, act: nn.Module = torch.tanh, dvc: str|torch.device = 'cpu',
            theta_i: list = []) -> RNN:
    """Generate an tanh RNN with random plus low-rank connectivity J + mn^T.

    Args:
        N: Dimension of the state space.
        K: Rank of the low-rank component of the connectivity.
        g: Scaling factor for the random connectivity.
        dvc: Device to put the models on.
        theta_i: expectation of n^{\\top}J^{i}m for i = 0, 1, ... of each low-rank
            component mn^{\\top}.
    
    Returns:
        mdl: an RNN with random plus low-rank connectivity J + P.
    
    See https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013111
    for details. Generally speaking, the eigenspectrum of W lies uniformly within a
    circle centered at the origin with radius g. Besides, due to the existence of the
    low-rank component there will also be several real eigenvalues that could be outside
    the circle. If theta_i[i] = 0 for all i > 0 (which is typically the case), the
    only one such eigenvalue has an expectation of theta_i[0]. If theta_i is not set,
    this eigenvalue will be small and within the unit circle. The paper showed that
    the model will have a pair of nontrivial attractors if there are one real eigenvalue
    outliers.

    Remember that the Jacobian of the model at x = 0 is Wf'(0) - Id. For tanh, f'(0) = 1,
    so the eigenvalues of the Jacobian is that of W shifted by -1. Therefore, to make the
    origin unstable, set theta_i[0] to be larger than 1.
    """
    J = torch.randn(N, N) * (g / torch.sqrt(torch.Tensor([N])))
    W = J
    for k in range(K):
        m = torch.randn(N)
        if len(theta_i):
            n = torch.zeros_like(m)
            Jim = m
            for (i, theta) in enumerate(theta_i):
                n += theta / (g ** (2 * i)) * Jim
                Jim = J @ Jim
            n /= N
        else:
            n = torch.randn(N) / N
        W += torch.outer(m, n)
    D = torch.ones(N, dtype=torch.float32)
    mdl = RNN(N, act, W.to(dvc), D.to(dvc))
    return mdl


# %% Functions to generate test GRUs

def get_all_GRUs(f : str = 'GRU2D.csv') -> list[cGRU]:
    """load GRU parameters from a csv file
    
    Note: in the supplementary figures of the paper, the system ix
    and x were plotted in the wrong order. Also, we cannot replicate
    system xxxiv. We doubt that the first four parameters should be 
    (2 0 0 2) instead of (1 0 0 1) accroding to its similarity with
    system xxxvi, so we change it. Note that system xxxiv is undergoing
    bifurcation and will be excluded from our further analysis anyway.
    """
    import pandas as pd
    df = pd.read_csv(f)
    Uh = torch.Tensor(df[['Uh11', 'Uh12', 'Uh21', 'Uh22']].values).reshape(36, 2, 2)
    Ur = torch.Tensor(df[['Ur11', 'Ur12', 'Ur21', 'Ur22']].values).reshape(36, 2, 2)
    bh = torch.Tensor(df[['bh1', 'bh2']].values)
    br = torch.Tensor(df[['br1', 'br2']].values)
    return [cGRU(Uh[i], Ur[i], bh[i], br[i]) for i in range(len(Uh))]


# %% Functions to generate test MINDy models

def get_all_MINDys(f : str = 'data/MINDy100.mat', n : int = 0) -> list[MINDy]:
    """load MINDy parameters from a mat file

    The models came from the original MINDy paper but for 100 instead of 400 nodes.
    The data file is a (53 subjects, 2 sessions) cell array, each containing a
    structure with fields 'W', 'alpha', 'D'. We will return a 2D numpy array of
    MINDy models.
    """
    allMdl = loadmat(f)['allMdl']
    sz = allMdl.shape
    allMdl = allMdl.flatten()
    allMdl = [MINDy(W=torch.tensor(mdl['W'][0, 0], dtype=torch.float32),
                    alpha=torch.tensor(mdl['alpha'][0, 0], dtype=torch.float32),
                    D=torch.tensor(mdl['D'][0, 0], dtype=torch.float32)) for mdl in allMdl]
    allMdl = np.array(allMdl).reshape(sz)
    if n > 0:
        allMdl = allMdl[:n]
    return allMdl


# %% Functions to generate systems restricted to lower-dimensional manifolds learned through DFORM

def get_restricted_model(full_mdl: Model, h: Model, dim: int = 2) -> Model:
    """Get a model that is restricted to the first `dim` dimensions of the full model"""
    return Restricted(Transformed(full_mdl, h), list(range(dim)))


# %% Some tests

if __name__ == '__main__':
    for _ in range(10):
        get_good_cond_matrix(5)
        get_good_cond_matrix(20)
        get_good_cond_matrix(200)
    for _ in range(10):
        eig_real = torch.randn(15)
        eig_imag = torch.cat([torch.randn(14), torch.zeros(1)])
        eig_real[1:-1:2], eig_imag[1:-1:2] = eig_real[:-1:2], -eig_imag[:-1:2]
        res = torch.linalg.eigvals(eig2real(eig_real, eig_imag))
        assert torch.allclose(torch.real(res).sort()[0], eig_real.sort()[0]) and \
            torch.allclose(torch.imag(res).sort()[0], eig_imag.sort()[0])
    for _ in range(10):
        A1, A2, H = get_similar_LTIs(6, [1, 1, 2], [0, 2, 0])
        print('Expecting one positive, one negative, two zeros, and two on left half plane')
        print(A1.A)
        print(torch.linalg.eigvals(A1.A))
        print(A2.A)
        print(torch.linalg.eigvals(A2.A))
    