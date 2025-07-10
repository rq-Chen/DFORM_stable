"""DFORM: diffeomorphic vector field alignment network
"""


# %% Dependencies

import torch
import torch.nn as nn
from torch.autograd.functional import jvp
from torchdiffeq import odeint, odeint_adjoint
from torch.nn.functional import cosine_similarity
from torch.linalg import vector_norm
from sklearn.decomposition import PCA
import numpy as np
import warnings
from typing import Callable, Any, Tuple, Sequence
from torch import Tensor
from matplotlib import pyplot as plt
from visualizations import PlotVec

# For debugging
import logging


# %% Utility functions

def myJvp(func: Callable[[Any], Tensor], inputs: Tuple, v: Tuple) -> Tuple[Tensor, Tensor]:
    """A wrapper of jvp with create_graph=True"""
    return jvp(func, inputs, v, create_graph=True)

def my_odeint(func: Callable[[Any], Tensor], y0: Tensor, t: float | int | Tensor,
              adjoint: bool = False, **kwargs) -> Tensor:
    """A wrapper of odeint to compute all or only the final value of the solution
    
    This wrapper does several things:

    1. If `t` is a scalar or a tensor with shape [] or (1,), we will assume t0 = 0.
    2. If `t` has no more than two elements, we will return the final state y(t) only,
        which will have the same shape as y0.
    3. It has an additional argument `adjoint` to use `odeint_adjoint` instead of `odeint`.
    4. If not specified, it will set the relative tolerance to 1e-4 instead of the default 1e-7.
    5. If the first call fails, it will fall back to `rk4` solver with a step size of 1/100 of
        the range of t.

    Otherwise the behavior is the same as `odeint` or `odeint_adjoint`.
    """
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-4
    if isinstance(t, (float, int)):
        t = torch.tensor(float(t), device=y0.device)
    if t.ndim == 0:
        t = torch.stack([torch.zeros_like(t), t])
    elif t.ndim == 1 and t.shape[0] == 1:
        t = torch.cat([torch.zeros_like(t), t])
    
    try:
        if adjoint:
            ans = odeint_adjoint(func, y0, t, **kwargs)
        else:
            ans = odeint(func, y0, t, **kwargs)
    except Exception as e:
        warnings.warn(f'odeint failed: {e}. Falling back to rk4 solver with a step size of 1/100 of the range of t.')
        step_size = (t.max() - t.min()) / 100
        if adjoint:
            ans = odeint_adjoint(func, y0, t, method='rk4', options={'step_size': step_size})
        else:
            ans = odeint(func, y0, t, method='rk4', options={'step_size': step_size})
    
    return ans[1] if len(t) <= 2 else ans

class Model(nn.Module):
    """A wrapper of nn.Module that allows easy access to device, dimensionality and samples

    Models should subclass this and implement the `get_samples` method to generate samples.
    """
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__()
        self.dim = dim
        self.device = torch.device('cpu')
    def to(self, device: str | torch.device) -> nn.Module:
        self.device = device
        super().to(device)
        return self
    def cpu(self) -> nn.Module:
        return self.to('cpu')
    def cuda(self) -> nn.Module:
        return self.to('cuda')
    def get_samples(self, n: int, *args, **kwargs) -> Tensor:
        return torch.randn(n, self.dim, device=self.device)

class Scaling(nn.Module):
    """A simple scaling layer"""
    def __init__(self, c: float = 0.) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(c, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x
   

# %% Linear transformation model

class MyLinear(Model):
    """A linear transformation model

    The forward method computes y = Wx + b, where W is a learnable matrix and b is a learnable bias.
    The inverse method computes x = W^{-1}(y - b).
    """
    def __init__(self, dim: int, W: Tensor | None = None, b: Tensor | None = None, id_init: bool = False) -> None:
        super().__init__(dim)
        if W is not None:
            self.W = nn.Parameter(W)
        else:
            if id_init:
                self.W = nn.Parameter(torch.eye(dim, dtype=torch.float32))
            else:
                self.W = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32) * torch.sqrt(torch.tensor(1 / dim, dtype=torch.float32)))
        if b is not None:
            self.b = nn.Parameter(b)
        else:
            self.b = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W.T + self.b
    def inverse(self, y: Tensor) -> Tensor:
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        return torch.linalg.solve(self.W.T, y - self.b, left=False)
    def get_samples(self, n: int) -> Tensor:
        return self(torch.randn(n, self.dim, device=self.device))


# %% Neural ODE model for deformation

class DFORM(Model):
    """a time-varying vector field v(t, x) modeling the infinitesimal deformation.

    Given a vector field f(x), the point x0 will be mapped to the point x(1),
    where x(s) is the solution of dx/ds = v(s, x(s)) with x(0) = x0. In other words,
    the point x0 is mapped to the flow of v(t, x) from t=0 to t=1 evalueated at x0.
    
    In principal v(t, x) can be any neural network, but here we use a simple MLP. The
    default size of the MLP is (n + 1, max(2n + 2, 20), max(2n + 2, 20), n), where n
    is the dimension of the phase space. The first input dimension is for time t.
    The forward method of the class computes v(t, x). The "deform" method computes
    the flow from time 0 to 1, i.e., the deformation. The "inv_deform" method
    computes the inverse deformation.

    We also allow computing the regularization of the deformation for potential regularization.
    Instead of using the regularization of an RKHS, here we implement a rather simple L2 norm
    based on a given measure `samp_fn`.

    We provide an option to make the vector field time-invariant or time-varying.

    Now we add another linear transformation in front of the nonlinear layers to make
    it easier to learn a "big" deformation. The linear part and nonlinear part can
    be frozen separately during training.
    """
    def __init__(self, dim: int, n_hid: Sequence[int] = [], act_fn: str = 'ELU', include_linear: bool = True,
                 id_init_linear: bool = False, id_init_nonlinear: bool = True,
                 time_varying: bool = False, samp_fn: Callable[[Any], Tensor] | None = None, samp_size: int = 128,
                 W: Tensor | None = None, b: Tensor | None = None) -> None:
        """Initialize the DFORM model

        Arguments:
            - dim: dimension of the phase space
            - n_hid: number of hidden units in each hidden layer. Sequence of integers.
            - act_fn: name of the activation function.
            - include_linear: whether to include a linear layer in the model.
            - id_init_linear: whether to initialize the linear transformation as identity.
            - id_init_nonlinear: whether to add a learnable scalar to the output that is
                initialized to zero. This will make the initial nonlinear deformation the
                identity function (no deformation).
            - time_varying: whether the vector field is time-varying.
            - samp_fn: function to sample the phase space to calculate the norm.
                Should take one argument (samp_size). Default to standard normal.
            - samp_size: number of samples to calculate the norm.
            - W: initial value of the linear transformation matrix. If None, will be random.
            - b: initial value of the bias. If None, will be zero.
        """
        super().__init__(dim)
        self.n_input = time_varying + dim  # Time is the first input (if time_varying)
        self.n_hid = n_hid if len(n_hid) else [np.maximum(20, self.n_input * 2) for _ in range(2)]
        self.n_output = dim
        self.act_fn = getattr(nn, act_fn)()
        self.include_linear = include_linear
        self.id_init_linear = id_init_linear
        self.id_init_nonlinear = id_init_nonlinear
        self.time_varying = time_varying
        self.samp_fn = samp_fn
        self.samp_size = samp_size

        # Linear layer
        if not self.include_linear:
            self.linear = nn.Identity()
        else:
            self.linear = MyLinear(dim, W, b, id_init_linear)

        # Nonlinear layers
        all_dims = [self.n_input] + list(self.n_hid) + [self.n_output]
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:  # No act_fn after last layer to make sure R^n -> R^n
                layers.append(self.act_fn)
        if id_init_nonlinear:
            layers.append(Scaling())
        self.layers = nn.Sequential(*layers)

        # Generate samples for the norm calculation
        self.samples = self.get_samples(samp_size)
    
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute the vector field at time t and position x
        
        Note: t only needs to have compatible shape with x[..., [0]], e.g., if x
        is (4, 3, 2), t can be scalar, (1, 1), (3, 1), (4, 1, 1), etc., but not (1,)
        """
        # need the bracket to keep the last dimension; and can't use expand_as
        # because myJvp(odeint(forward)) will fail
        if self.time_varying:
            t = t * torch.ones_like(x[..., [0]])
            return self.layers(torch.cat([t, x], dim=-1))
        else:
            return self.layers(x)
    
    def flow(self, x0: Tensor, t: float | int | Tensor, **kwargs) -> Tensor:
        """Compute the flow of the vector field

        t should be either a scalar (end time) or a 1D tensor (time points,
        with the first element being start time). If t has no more than two
        elements, the function will return the final state which has the same
        shape as x0. Otherwise, it will return the whole trajectory with
        shape (len(t), *x0.shape), including t0. See `my_odeint` for more
        details. This behavior is different from the forward method, where
        t should be broadcastable to x[..., [0]] and output shape is the same
        as x.
        """
        return my_odeint(self, x0, t, **kwargs)

    def deform(self, x0: Tensor, **kwargs) -> Tensor:
        """Compute the deformation of the vector field"""
        if hasattr(self, 'only_linear') and self.only_linear:
            return self.linear(x0)
        elif not self.include_linear:
            return self.flow(x0, torch.tensor([0., 1.], device=self.device), **kwargs)
        else:
            return self.flow(self.linear(x0), torch.tensor([0., 1.], device=self.device), **kwargs)
    
    def inv_deform(self, x1: Tensor, **kwargs) -> Tensor:
        """Compute the inverse deformation of the vector field"""
        if hasattr(self, 'only_linear') and self.only_linear:
            return self.linear.inverse(x1)
        elif not self.include_linear:
            return self.flow(x1, torch.tensor([1., 0.], device=self.device), **kwargs)
        else:
            return self.linear.inverse(self.flow(x1, torch.tensor([1., 0.], device=self.device), **kwargs))

    def get_samples(self, n: int, *args, **kwargs) -> Tensor:
        """Generate samples for the norm calculation"""
        if self.samp_fn is None:
            return torch.randn(n, self.dim, device=self.device)
        else:
            return self.samp_fn(n, *args, **kwargs).to(self.device)
    
    def update_samples(self, samples: Tensor | None = None) -> Tensor:
        """Update the samples used to calculate the regularization of transformation
        
        If samples is None, we will generate new samples using `get_samples`. Otherwise
        the samples will be updated to the given tensor.
        """
        if samples is None:
            self.samples = self.get_samples(self.samp_size)
        else:
            self.samples = samples.to(self.device)
        return self.samples

    def SqNormT(self, t: Tensor, *args, **kwargs) -> Tensor:
        """Compute the squared regularization of the vector field at time t

        The unused parameters are for compatibility with `odeint`.

        In principal you can implement other norms, such as RKHS norm.
        """
        if hasattr(self, 'only_linear') and self.only_linear:
            return torch.tensor(0., device=self.device)        
        v = self(t, self.samples)
        return torch.sum(v ** 2, dim=-1).mean()
    
    def SqNorm(self, x: Tensor | None = None) -> Tensor:
        """Compute the squared norm of the deformation"""
        if hasattr(self, 'only_linear') and self.only_linear:
            return torch.tensor(0., device=self.device)
        self.update_samples(x)
        if self.time_varying:
            return my_odeint(self.SqNormT, torch.tensor((0.,), device=self.device),
                            torch.tensor([0., 1.], device=self.device))
        else:
            return self.SqNormT(torch.tensor(0., device=self.device))
    
    def freeze(self, part: str = 'all', switch: bool = True) -> None:
        """Freeze/unfreeze the linear or/and nonlinear parts of the model
        
        Arguments:
            - part: which part to freeze/unfreeze. Can be either 'linear', 'nonlinear' or 'all'.
            - switch: if False, unfreeze selected weights instead of freeze.
        """
        if part in ['all', 'linear']:
            for p in self.linear.parameters():
                p.requires_grad_(not switch)
        if part in ['all', 'nonlinear']:
            for p in self.layers.parameters():
                p.requires_grad_(not switch)
    
    def linear_pretrain(self, switch: bool = True) -> None:
        """Switch on/off the linear pretraining

        When switched on for the first time, if the nonlinear part is initialized as
        identity, a flag `only_linear` will be set to True. Under this flag, the deform and
        inv_deform methods will only use the linear part.
        """
        self.freeze('nonlinear', switch=switch)
        if switch and self.id_init_nonlinear and not hasattr(self, 'only_linear'):
            self.only_linear = True
        else:
            self.only_linear = False


# %% Draw samples from noisy simulations

class Sampler:
    """Draw samples from a dynamical system.

    Args:
        f: A dynamical system.
        x0: [N, D] initial states, where D is the dimension of f.
        noise_std: Standard deviation of the Gaussian noise.
        dt: Time step.
        T: Simulation time to obtain asymptotic distribution
        updating: See below.

    Will simulate system f from x0 for T/dt time steps and store the final states
    as the asymptotic distribution. When required to draw samples, if `updating`
    is False, samples will be directly drawn from the asymptotic distribution. If
    `updating` is True, we will simulate the system for one more step and update
    the saved samples before returning them. Doing so can ensure that the samples
    never repeat.

    IMPORTANT: if `updating` is true, we will call `f` each time we draw samples,
    which requires `f` to remain unchanged. Usually this should be the case because
    `f` is supposed to represent a time-invariant system.

    Also note that please move `f` to the device it will be on during training before
    constructing this class. `x0` also need to be on the same device as `f`.

    Besides, sometimes nan or inf will occur during initialization or sampling due to
    singularities or numerical integration error. We will remove these samples, so the
    number of samples might change slightly from batch to batch.
    """
    def __init__(self, f: Callable[[Tensor], Tensor], x0: Tensor,
            noise_std: float = 0.1, dt: float = 0.01, T: float = 100.,
            updating: bool = True) -> None:
        
        nan_thres = 0.1  # If more than this portion of samples are NaN, raise error

        with torch.no_grad():
            self.f = f
            self.noise_std = noise_std
            self.dt = dt
            self.updating = updating
            if hasattr(f, 'device'):
                self.device = f.device
            else:
                self.device = x0.device
            self.samples = x0.clone().detach().to(self.device)
            self.N, self.D = self.samples.shape
            
            # Simulate system
            n_steps = int(T / dt)
            for _ in range(n_steps):
                noise = torch.randn(self.N, self.D, device=self.device) * noise_std
                self.samples += (self.f(self.samples) + noise) * dt
            
            # Remove nan or inf samples
            nan_idx = torch.isnan(self.samples).any(dim=-1)
            if nan_idx.any():
                warnings.warn(f'Found {nan_idx.sum()}/{self.N} NaN samples!')
                if nan_idx.sum() / self.N > nan_thres:
                    raise RuntimeError(f'More than {nan_thres * 100}% of samples are NaN! Please check whether the system is bounded.')
                else:
                    warnings.warn('Could be numerical integration error. Removing NaN samples...')
                    self.samples = self.samples[~nan_idx]
                    self.N = self.samples.shape[0]
    
    def __call__(self, n: int, *args) -> Any:
        """Get n samples from the asymptotic distribution.

        Args:
            n: Number of samples.
        """

        with torch.no_grad():
            idx = torch.randint(0, self.N, (n,), device=self.device)
            noise = torch.randn(n, self.D, device=self.device) * self.noise_std
            samples = self.samples[idx] + (self.f(self.samples[idx]) + noise) * self.dt
            # Remove nan or inf samples
            idx_bad = torch.logical_or(torch.any(torch.isnan(samples), dim=-1),
                                       torch.any(torch.isinf(samples), dim=-1))
            if idx_bad.sum():
                warnings.warn(f'Removing {idx_bad.sum()}/{n} bad samples.')
            idx_good = torch.logical_not(idx_bad)
            if self.updating:
                self.samples[idx[idx_good]] = samples[idx_good]
            return samples[idx_good].clone().detach()

    def to(self, device: str | torch.device):
        self.device = device
        self.samples = self.samples.to(device)
        try:
            self.f = self.f.to(device)
        except:
            warnings.warn('Cannot move Sampler.f, ignoring...')
        return self

    def cpu(self):
        return self.to('cpu')
    
    def cuda(self):
        return self.to('cuda')


# %% Loss functions and utilities to handle different dimensionality

class MyLoss(nn.Module):
    def __init__(self, lossFn: str = 'cosine', normalized: bool = False) -> None:
        super().__init__()
        assert lossFn in ['cosine', 'MSE'], 'lossFn must be either cosine or MSE'
        self.lossFn = lossFn
        self.normalized = normalized
        self.mse = nn.MSELoss()
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # Compare only the first min_dim elements
        min_dim = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :min_dim], b[..., :min_dim]
        if self.lossFn == 'cosine':
            return 1 - torch.mean(cosine_similarity(a, b, dim=-1))
        elif self.lossFn == 'MSE':
            if self.normalized:
                return self.mse(a / vector_norm(a, dim=-1, keepdim=True),
                    b / vector_norm(b, dim=-1, keepdim=True))
            else:
                return self.mse(a, b)
        else:
            raise ValueError('lossFn must be either cosine or MSE')

def adjust_dim(x: Tensor, dim: int, method: str = "zeros") -> Tensor:
    """Adjust the dimensionality of x to dim"""
    old_dim = x.shape[-1]
    if old_dim >= dim:
        return x[..., :dim]
    pad_shape = x.shape[:-1] + (dim - old_dim,)
    if method == "zeros":
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)
    elif method == "randn":
        pad = (torch.randn(pad_shape, dtype=x.dtype, device=x.device) - torch.mean(x)) * torch.std(x)
        return torch.cat([x, pad], dim=-1)

def link_PCs(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    """Link the principal components of x and y

    Args:
        x: [N, D1] samples from the source system
        y: [N, D2] samples from the target system (D1 >= D2)
    
    Returns:
        W: [D1, D1] matrix
        b: [D1] bias term
    
    The last D1 - D2 columns of W will be orthogonal to the first D2 columns.

    The idea is to convert f's samples to scores on the principal components,
    then use these scores and g's principal components to construct samples
    in g's space. This method may help when using asymptotic samples, because
    the first several PCs are usually determined by the stable fixed points
    and limit cycles.
    """

    # Use PCA to link the principal components
    pca_f = PCA().fit(x.cpu().numpy())
    pca_g = PCA().fit(y.cpu().numpy())
    D1, D2 = x.shape[-1], y.shape[-1]
    f_coeffs = pca_f.components_.T  # [D1, n_components (D1)]
    f_bias = pca_f.mean_  # [D1]
    # f_score = (x - f_bias) @ f_coeffs  # [N, n_components]
    g_comps = pca_g.components_  # [n_components (D2), D2]
    g_bias = pca_g.mean_  # [D2]
    g_comps_full = np.eye(D1)
    g_comps_full[:D2, :D2] = g_comps
    g_bias_full = np.zeros(D1)
    g_bias_full[:D2] = g_bias
    # y = g_score @ g_comps + g_bias  # [N, D2]
    # y_full = [y 0] = [g_score 0] @ [g_comps 0; 0 Id] + [g_bias 0]  # [N, D1]
    # [y 0] = (x - f_bias) @ f_coeffs @ g_comps_full + g_bias_full
    W = (f_coeffs @ g_comps_full).T  # In linear layer we use y = x @ W.T + b
    b = -f_bias @ W.T + g_bias_full

    return torch.tensor(W, dtype=x.dtype, device=x.device), torch.tensor(b, dtype=x.dtype, device=x.device)


# %% Training

def Training(f: Model, g: Model, f_samp: Callable | None = None, g_samp: Callable | None = None,
             asymp_samp_noise: Tuple[float | None, float | None] = [None, None], adjust_dim_method: str = "zeros",
             pca_init_noise: float | None = None, samp_domain_W=0., linear_regW=0., regW=0., g2fW=1., lossFn='MSE', warpTime=True, 
             linear_lr=2e-3, lr=2e-3, nBatch_linear=100, nBatch=100, batch_size=128, earlyStopThres=0,
             vis=True, verbose=True, Sampler_kw={}, **kwargs):
    """Training DFORM to match the topology or geometry of two vector fields
    
    Args:
        `f` (Model): The source vector field
        `g` (Model): The target vector field, should have same or lower dimensionality than `f`
        `f_samp` (Callable | None): A function that generates samples from `f`'s domain. If None,
            samples will be drawn from `f.get_samples()` or asymptotic distribution with noise.
        `g_samp` (Callable | None): A function that generates samples from `g`'s domain. If None,
            samples will be drawn from `g.get_samples()` or asymptotic distribution with noise.
        `asymp_samp_noise` (float | None * 2): If `f_samp` and/or `g_samp` is None and `asymp_samp_noise`
            is not None, samples for `f` and/or `g` will be drawn from asymptotic distributions of
            noisy simulations of `f` and/or `g` with the this noise standard deviation.
        `adjust_dim_method` (str): Method to adjust the dimensionality of the samples. Only matters
            if f.dim != g.dim. Can be either "zeros" or "randn" (see docstring of `adjust_dim`).
        `pca_init_noise` (float | None): If zero, the linear part will be initialized by linking the
            PCs of the samples drawn from `f.get_samples()` and `g.get_samples()`. If positive, it uses
            the samples drawn from the asymptotic distributions of noisy simulations of `f` and
            `g` with this noise standard deviation. If none, initialized randomly.
        `samp_domain_W` (float): Weight of the loss calculated by sampling from h or h_inv's domain
        `linear_regW` (float): Weight of the regularization term for the norm of WW^T - I, where W is the
            linear transformation matrix.
        `regW` (float): Weight of the regularization term for the norm of phi - Id, where phi is the
            nonlinear part of the deformation.
        `g2fW` (float): Weight of the loss from g to f. Set to 0 to align f to g only.
        `lossFn` (str): Loss function, either `MSE` or `cosine`
        `warpTime` (bool): Whether to warp the time dimension (i.e., normalize the speed)
        `linear_lr` (float): Learning rate for training the linear part
        `lr` (float): Learning rate for training the whole model
        `nBatch_linear` (int): Number of training batches for the linear part.
        `nBatch` (int): Number of training batches for the whole model
        `batch_size` (int): Batch size
        `earlyStopThres` (float): Stop training if loss decreases in `early_stop_len` batches
            is less than this ratio. Default to 0, which means no early stopping.
        `vis` (bool): Whether to visualize the training loss
        `verbose` (bool): Whether to print the training loss
        `Sampler_kw` (dict): Keyword arguments for `Sampler`
        `**kwargs`: Keyword arguments for `DFORM`, e.g., `n_hid`, `include_linear`
    
    Returns:
        `mdl`: a DFORM that maps from the f's domain to the g's domain
        `f_samp`: Either f.get_samples() or a `Sampler` object that samples from f's domain
        `g_samp`: Either g.get_samples() or a `Sampler` object that samples from g's domain
        `losses`: a tensor of shape (nBatch, 5) that records the loss at each batch: Lie loss
            sampled from the codomain, sampled from domain, regularization for linear part,
            regularization for nonlinear part, and total loss
    """

    assert lossFn in ['MSE', 'cosine'], 'lossFn must be either MSE or cosine'
    assert warpTime or lossFn == 'MSE', 'warpTime must be True if lossFn is cosine'

    loss_avg_w = 0.9  # AR(1) loss moving average weight
    early_stop_len = 100  # Compare current loss with loss from this many batches ago (for early stopping)
    asymp_n_smp = 4096  # Number of samples for asymptotic sampling

    # Initialize models
    mdl = DFORM(f.dim, **kwargs)
    dvc = f.device
    mdl.to(dvc)

    # Loss function for pushforward
    LieLoss = MyLoss(lossFn, normalized=warpTime)
    
    # Keep track of losses
    losses = torch.zeros(nBatch_linear + nBatch, 5)  # Lie derivative loss (codomain, domain), regularization term, total loss
    avg_loss = torch.zeros(nBatch_linear + nBatch)  # Smoothed total loss

    # Small utility function to get the sampling function
    def set_samp(f: Model, f_samp: Callable | None, noise: float | None, **kwargs) -> Callable:
        if f_samp is not None:
            print('Using provided sampling function to calculate loss.')
        elif noise is not None:
            assert noise >= 0, 'Noise standard deviation must be non-negative!'
            print('Using noisy asymptotic distributions to calculate loss.')
            print(f'Simulating with noise std = {noise} ...', end=' ')
            try:
                f_samp = Sampler(f, f.get_samples(asymp_n_smp).to(f.device), noise, **kwargs)
                print('done.')
            except:
                warnings.warn('Failed to sample asymptotic distribution. ' +
                            'Please check if the system is bounded!')
                warnings.warn('Falling back to sampling from the original distribution ...')
                f_samp = f.get_samples
        else:
            print('Use get_samples() method to generate samples to calculate loss.')
            f_samp = f.get_samples
        return f_samp

    # Functions to sample from the old and target domains
    if not isinstance(asymp_samp_noise, Sequence):
        asymp_samp_noise = [asymp_samp_noise, asymp_samp_noise]
    print('Selecting sampling functions for system f ...')
    f_samp = set_samp(f, f_samp, asymp_samp_noise[0], **Sampler_kw)
    print('Selecting sampling functions for system g ...')
    g_samp = set_samp(g, g_samp, asymp_samp_noise[1], **Sampler_kw)
    
    # Initialize the linear part with PCA
    if not mdl.include_linear:
        print('Linear layer not included, skipping initialization ...')
    elif pca_init_noise is None:
        if "id_init_linear" in kwargs and kwargs["id_init_linear"]:
            print('Initializing the linear part as identity ...')
        else:
            print('Initializing the linear part randomly ...')
    else:
        assert pca_init_noise >= 0, 'Noise standard deviation must be non-negative!'
        if pca_init_noise > 0:
            print(f'Initializing the linear part with PCA of noisy simulations with noise std = {pca_init_noise} ...', end=' ')
            if pca_init_noise == asymp_samp_noise:  # Which is often the case
                print('Same noise level, using the same sampling functions.', end=' ')
                pca_f_samp = f_samp
                pca_g_samp = g_samp
            else:
                try:
                    pca_f_samp = Sampler(f, f.get_samples(asymp_n_smp), pca_init_noise, **Sampler_kw)
                    pca_g_samp = Sampler(g, g.get_samples(asymp_n_smp), pca_init_noise, **Sampler_kw)
                except:
                    warnings.warn('Failed to sample asymptotic distribution for PCA initialization. ' +
                                'Please check if the system is bounded!')
                    warnings.warn('Falling back to sampling from the original distribution ...')
                    pca_f_samp = f.get_samples
                    pca_g_samp = g.get_samples
        elif pca_init_noise == 0:
            print('Initializing the linear part with PCA of the samples drawn from the systems ...', end=' ')
            pca_f_samp = f_samp
            pca_g_samp = g_samp
        f_samples = pca_f_samp(asymp_n_smp)
        g_samples = pca_g_samp(asymp_n_smp)
        W, b = link_PCs(f_samples, g_samples)
        mdl.linear.W.data = W
        mdl.linear.b.data = b
        print('done.')

    ##########################  One training batch  ################################

    def one_batch(optimizer: torch.optim.Optimizer, loss_output: Tensor | None = None, avg_loss_output: Tensor | None = None,
                  i_batch: int | None = None, n_batch: int | None = None) -> Tensor:
        """Run one batch"""

        codomain_loss, domain_loss, Wreg_norm, deform_norm, loss = torch.zeros(5, device=dvc)

        # Sample from g's domain
        y_in_g = g_samp(batch_size).to(dvc)
        y = adjust_dim(y_in_g, mdl.dim, adjust_dim_method)  # Append zeros or random entries
        if samp_domain_W > 0 and g2fW > 0:
            gy = adjust_dim(g(y_in_g), mdl.dim, adjust_dim_method)
            x, Jinvgy = myJvp(mdl.inv_deform, (y,), (gy,))
            fx = f(x)
            domain_loss = domain_loss + g2fW * LieLoss(fx, Jinvgy)  # mismatch between f and hinv_*g sampled from g's domain
        else:
            x = mdl.inv_deform(y)
            fx = f(x)
        hx, Jfx = myJvp(mdl.deform, (x,), (fx,))
        hx_in_g = adjust_dim(hx, g.dim, adjust_dim_method)  # Truncated to g's dimension (use y_in_g might also work)
        codomain_loss = codomain_loss + LieLoss(Jfx, g(hx_in_g))  # Compare only the first g.dim elements

        # Sample from f's domain
        if g2fW > 0 or samp_domain_W > 0:
            x = f_samp(batch_size).to(dvc)
            if samp_domain_W > 0:
                fx = f(x)
                y, Jfx = myJvp(mdl.deform, (x,), (fx,))
                y_in_g = adjust_dim(y, g.dim, adjust_dim_method)
                gy_in_g = g(y_in_g)
                domain_loss = domain_loss + LieLoss(Jfx, gy_in_g)  # mismatch between (first g.dim elements of) h_*f and g sampled from f's domain
            else:
                y = mdl.deform(x)
                y_in_g = adjust_dim(y, g.dim, adjust_dim_method)
                gy_in_g = g(y_in_g)
            if g2fW > 0:
                gy = adjust_dim(gy_in_g, mdl.dim, adjust_dim_method)
                hinvy, Jinvgy = myJvp(mdl.inv_deform, (y,), (gy,))
                codomain_loss = codomain_loss + g2fW * LieLoss(f(hinvy), Jinvgy)  # mismatch between f and hinv_*g sampled from f's domain

        # Regularization
        if linear_regW > 0 and mdl.include_linear:
            Wreg_norm = torch.norm(mdl.linear.W @ mdl.linear.W.T - torch.eye(mdl.dim, device=dvc))
        else:
            Wreg_norm = torch.tensor([0.], device=dvc)
        if regW > 0:
            deform_norm = mdl.SqNorm()
        else:
            deform_norm = torch.tensor([0.], device=loss.device)
        
        # Update weights
        loss = codomain_loss + samp_domain_W * domain_loss + linear_regW * Wreg_norm + regW * deform_norm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if torch.isnan(loss).any():
            raise RuntimeError('NaN loss!')
        
        # Record loss
        curr_loss = torch.tensor([codomain_loss.item(), domain_loss.item(), Wreg_norm.item(), deform_norm.item(), loss.item()])
        if loss_output is not None:
            loss_output[i_batch] = curr_loss
        
        # Record average loss
        if i_batch == 0:
            last_loss = curr_loss[-1]
        else:
            last_loss = avg_loss_output[i_batch - 1]
        avg_loss_output[i_batch] = last_loss * loss_avg_w + (1 - loss_avg_w) * curr_loss[-1]
        
        # Print loss
        if i_batch is not None and n_batch is not None:
            if verbose or i_batch == n_batch - 1 or i_batch % 100 == 99:
                print(f'Batch {i_batch + 1}/{n_batch}, Lie loss sampled from codomain = {curr_loss[0]}, ' +
                      f'sampled from domain = {curr_loss[1]}, ' +
                    f'R(||WW^T - I||) = {curr_loss[2]}, R(|phi - Id|) = {curr_loss[3]}, total = {curr_loss[4]}')
        
        return curr_loss
    
    ########################### Train the linear part ################################
    
    if not mdl.include_linear:
        print('Linear layer not included, skipping linear pretraining ...')
    elif nBatch_linear > 0:

        # Record the reflection of the initial weights for the linear part
        W0_neg = -mdl.linear.W.clone().detach()
        b0_neg = -mdl.linear.b.clone().detach()
        if mdl.dim % 2 == 0:  # To enforce a negation of the determinant
            W0_neg[0, :] = -W0_neg[0, :]
            b0_neg[0] = -b0_neg[0]

        # Training only the linear part
        print('Training the linear part ...')
        optim_linear = torch.optim.NAdam(mdl.parameters(), lr=linear_lr)
        loss_linear = torch.zeros(nBatch_linear, 5)
        avg_loss_linear = torch.zeros(nBatch_linear)
        mdl.linear_pretrain(switch=True)
        for i in range(nBatch_linear):
            one_batch(optimizer=optim_linear, loss_output=loss_linear,
                        avg_loss_output=avg_loss_linear, i_batch=i, n_batch=nBatch_linear)

        # Try training and testing the linear part with the reflection of initial weights
        # This helps the model to deal with reflections.
        print('Training the linear part with the reflection of initial weights ...')
        W = mdl.linear.W.clone().detach()
        b = mdl.linear.b.clone().detach()
        mdl.linear.W.data = W0_neg
        mdl.linear.b.data = b0_neg
        optim_linear = torch.optim.NAdam(mdl.parameters(), lr=linear_lr)
        loss_linear_neg = torch.zeros(nBatch_linear, 5)
        avg_loss_linear_neg = torch.zeros(nBatch_linear)
        for i in range(nBatch_linear):
            one_batch(optimizer=optim_linear, loss_output=loss_linear_neg,
                        avg_loss_output=avg_loss_linear_neg, i_batch=i, n_batch=nBatch_linear)

        # Pick the one with lower loss
        if avg_loss_linear_neg[-1] < avg_loss_linear[-1]:
            losses[:nBatch_linear] = loss_linear_neg
            avg_loss[:nBatch_linear] = avg_loss_linear_neg
        else:
            mdl.linear.W.data = W
            mdl.linear.b.data = b
            losses[:nBatch_linear] = loss_linear
            avg_loss[:nBatch_linear] = avg_loss_linear
    
    # Test the linear part
    print('Testing the model with only linear transformation ...')
    Testing(mdl, f, g, f_samp, g_samp, adjust_dim_method, vis=False)

    ######################### Train the whole model #################################

    # Training the whole model
    mdl.linear_pretrain(switch=False)
    print('Training the whole model ...')
    optim = torch.optim.NAdam(mdl.parameters(), lr=lr)

    for i in range(nBatch):

        one_batch(optimizer=optim, loss_output=losses[nBatch_linear:],
                  avg_loss_output=avg_loss[nBatch_linear:], i_batch=i, n_batch=nBatch)

        # Early stopping
        if earlyStopThres > 0 and i >= early_stop_len and i % 100 == 0:
            tmp = avg_loss[i + nBatch_linear] / avg_loss[i + nBatch_linear - early_stop_len]
            if tmp > 1 - earlyStopThres and tmp < 1:
                print(f'\nEarly stopping at batch {i + 1}!')
                print(f'Batch {i + 1}/{nBatch}, Lie loss sampled from codomain = {losses[i + nBatch_linear, 0]}, ' +
                      f'sampled from domain = {losses[i + nBatch_linear, 1]}, ' +
                    f'R(||WW^T - I||) = {losses[i + nBatch_linear, 2]}, ' +
                    f'R(|phi - Id|) = {losses[i + nBatch_linear, 3]}, total = {losses[i + nBatch_linear, 4]}')
                break

    print()
    if vis and (nBatch_linear + nBatch > 0):
        tmp = torch.cat([losses, avg_loss.unsqueeze(1)], dim=1)
        fig, ax1 = plt.subplots()
        ax1.plot(tmp[:, 4].numpy(), color='tab:blue')
        ax1.plot(tmp[:, -1].numpy(), color='tab:red')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.legend(['Weighted loss', 'Smoothed weighted loss'], loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(tmp[:, 2].numpy(), color='tab:green')
        ax2.plot(tmp[:, 3].numpy(), color='tab:orange')
        ax2.set_ylabel('Regularization')
        ax2.legend(['R(||WW^T - I||)', 'R(|phi - Id|)'], loc='upper right')
        plt.title('Training Loss')
        plt.show()
    
    # Testing the full model
    print('Testing the full model ...')
    Testing(mdl, f, g, f_samp, g_samp, adjust_dim_method, vis=vis)

    return mdl, f_samp, g_samp, losses


# %% Testing

def Testing(mdl, f, g, f_samp=None, g_samp=None, adjust_dim_method="zeros",
            vis=True, x_grid=np.linspace(-5, 5, 20), y_grid=np.linspace(-5, 5, 20),
            **kwargs):

    # constants
    batch_size = 300

    # cosine similarity for min_dim elements
    def mycos(a, b):
        min_dim = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :min_dim], b[..., :min_dim]
        return cosine_similarity(a, b, dim=-1).mean().item()

    try:
        dvc = f.device
    except:
        dvc = 'cpu'
    
    if f_samp is None:
        f_samp = f.get_samples
    if g_samp is None:
        g_samp = g.get_samples
    
    # 0. Between f & g, sampled with g_samp
    # 1. Between f & g, sampled with f_samp
    # 2. Between mdl_*f & g, sampled with g_samp
    # 3. Between mdl_*f & g, sampled with f_samp
    # 4. Between f & mdl_inv_*g, sampled with f_samp
    # 5. Between f & mdl_inv_*g, sampled with g_samp
    sims = [0.] * 6
    
    with torch.no_grad():

        ## sampling from g's domain ##

        # Difference between f and g, sampled with g_samp
        y_in_g = g_samp(batch_size).to(dvc)
        y = adjust_dim(y_in_g, mdl.dim, adjust_dim_method)  # Append zeros or random entries
        gy_in_g = g(y_in_g)
        fy = f(y)
        sims[0] = mycos(fy, gy_in_g)  # mycos() calculates cosine similarity for the first g.dim elements

        # Difference between f and mdl_inv_*g, sampled with g_samp
        gy = adjust_dim(gy_in_g, mdl.dim, adjust_dim_method)
        x, Jinvgy = jvp(mdl.inv_deform, (y,), (gy,))
        fx = f(x)
        sims[5] = mycos(fx, Jinvgy)

        # Difference between mdl_*f and g, sampled with g_samp
        hx, Jfx = jvp(mdl.deform, (x,), (fx,))
        hx_in_g = adjust_dim(hx, g.dim, adjust_dim_method)  # Truncated to g's dimension (use y_in_g might also work)
        ghx_in_g = g(hx_in_g)
        sims[2] = mycos(Jfx, ghx_in_g)

        ## sampling from f's domain ##

        # Difference between f and g, sampled with f_samp
        x = f_samp(batch_size).to(dvc)
        fx = f(x)
        x_in_g = adjust_dim(x, g.dim, adjust_dim_method)
        gx_in_g = g(x_in_g)
        sims[1] = mycos(fx, gx_in_g)

        # Difference between mdl_*f and g, sampled with f_samp
        y, Jfx = jvp(mdl.deform, (x,), (fx,))
        y_in_g = adjust_dim(y, g.dim, adjust_dim_method)  # Truncated to g's dimension
        gy_in_g = g(y_in_g)
        sims[3] = mycos(Jfx, gy_in_g)      

        # Difference between f and mdl_inv_*g, sampled with f_samp
        gy = adjust_dim(gy_in_g, mdl.dim, adjust_dim_method)
        hinvy, Jinvgy = jvp(mdl.inv_deform, (y,), (gy,))
        sims[4] = mycos(f(hinvy), Jinvgy)  

        # Output
        print("Cosine similarity between f and g without coordinate transformation, " +
                "averaged across g's measure: {:.4f}, ".format(sims[0]) +
                "across f's measure: {:.4f}".format(sims[1]))
        print("Cosine similarity between pushforward of f and g, " +
                "averaged across g's measure: {:.4f}, ".format(sims[2]) +
                "across f's measure: {:.4f}".format(sims[3]))
        print("Cosine similarity between f and inverse pushforward of g, " +
                "averaged across f's measure: {:.4f}, ".format(sims[4]) +
                "across g's measure: {:.4f}".format(sims[5]))

        # Visualize vector fields
        if vis and f.dim == 2 and g.dim == 2:

            f.to('cpu')
            g.to('cpu')
            mdl.to('cpu')

            # Visualize f(x) and its push-forward by mdl
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            PlotVec(f, x=x_grid, y=y_grid, f_samp_func=f_samp, **kwargs)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.title('$f$')

            plt.subplot(1, 3, 2)
            PlotVec(f, lambda x: jvp(mdl.deform, (x,), (f(x),)), x=x_grid, y=y_grid, f_samp_func=f_samp, **kwargs)
            plt.xlabel('$y_1$')
            plt.ylabel('$y_2$')
            xl = plt.xlim()
            yl = plt.ylim()
            plt.title('$mdl_{*}f$')
            
            plt.subplot(1, 3, 3)
            PlotVec(g, x=np.linspace(xl[0], xl[1], len(x_grid)), y=np.linspace(yl[0], yl[1], len(y_grid)),
                    f_samp_func=g_samp, color='r', alpha=0.5, **kwargs)
            plt.xlabel('$y_1$')
            plt.ylabel('$y_2$')
            plt.xlim(xl)
            plt.ylim(yl)
            plt.title('$g$')

            plt.suptitle('f(x) and its push-forward by mdl')
            plt.show()

            f.to(dvc)
            g.to(dvc)
            mdl.to(dvc)
    
    return sims


# %% Quantify similarity

def get_topo_sim(f, g, n_rep=1,
            **kwargs) -> Tuple[float, DFORM, Sampler | Callable, Sampler | Callable, Tensor, list[float]]:

    try:
        mdl, f_samp, g_samp, losses = Training(f, g, vis=False, verbose=False, **kwargs)
        sims = Testing(mdl, f, g, f_samp, g_samp, vis=False)
        sim = min(sims[2], sims[4])  # Use the smaller one - pushforward of f or inverse pushforward of g
        res = (sim, mdl, f_samp, g_samp, losses, sims)
    except Exception as e:
        # print(e)
        logging.exception(e)
        warnings.warn('Failed to train the model. Returning NaN...')
        res = (float('nan'), None, None, None, None, [float('nan')] * 6)
    
    if n_rep <= 1:
        return res
    else:
        res2 = get_topo_sim(f, g, n_rep=n_rep-1, **kwargs)
        if res[0] > res2[0] or res2[0] == float('nan'):
            return res
        else:
            return res2


# %% Some tests

if __name__ == '__main__':

    # An example with two equivalent systems
    # Takes around 5 minutes to run on an old CPU
    from example_systems import VanDerPol
    from visualizations import PlotField
    f = VanDerPol(0.2)
    g = VanDerPol(2)
    PlotField([f, g])
    cfg = {
        'nBatch': 3000,
        'nBatch_linear': 2000,
        'linear_lr': 2e-3,
        'lr': 2e-4,
        'batch_size': 32,
        'verbose': False,
    }
    mdl = Training(f, g, **cfg)[0]