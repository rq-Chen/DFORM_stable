"""visualizations.py - Functions for visualizing data."""

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from typing import Any, Sequence, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# %% 2D vector field visualization

def PlotField(fs, x=torch.linspace(-5, 5, 20), y=torch.linspace(-5, 5, 20), title=None):
    """Streamline plot of 2D vector fields"""
    nF = len(fs)
    if title is None:
        title = [f'Field {i}' for i in range(nF)]
    X, Y = torch.meshgrid(x, y, indexing='xy')
    fig, axs = plt.subplots(1, nF, sharex=True, sharey=True, figsize=(10, 6))
    if nF == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten()
    for i in range(nF):
        Fx = fs[i](torch.stack([X, Y], dim=-1))
        U, V = Fx.detach().clone().numpy()[..., 0], Fx.detach().clone().numpy()[..., 1]
        vfx = np.sqrt(U ** 2 + V ** 2)
        axs[i].streamplot(X.numpy(), Y.numpy(), U, V, color=vfx, linewidth=2, cmap='autumn')
        axs[i].set_title(title[i])
    plt.show()
    return fig, axs


def PlotVec(f, hxJfx=None, x=np.linspace(-5., 5., 20), y=np.linspace(-5., 5., 20),
            color=None, alpha=None, normlen=False, arrow_rescale=1.,
            f_samp_func=None, plt_traj=True, X0=None, T=20, step_per_T=20, traj_color='gray'):
    """Vector plot of a 2D vector field or its push-forward
    
    If `hxJfx` is provided, it should return a tuple hxJfx(x) = (h(x), J_h(x)f(x)) and the push-forward
    vector field will be plotted. Otherwise, the original vector field will be plotted.

    If `f_samp_func` is provided, will add a contour plot of the distribution represented by `f_samp_func`,
    or its push-forward if `f` returns a tuple.

    If `plt_traj` is True, will plot the trajectory of nine samples from 1/4, 2/4, 3/4 of the x and y ranges.

    Note: the vector field is plotted using the `arrow` function, so the arrow length and width depends on
    the aspect ratio of the plot.
    """

    if hxJfx is not None:
        is_push_forward = True
    else:
        is_push_forward = False

    # Number of samples to visualize the distribution
    n_samples = 2000
    # Simulate trajectories
    nSteps = T * step_per_T

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    X, Y = np.meshgrid(x, y, indexing='xy')
    X, Y = X.flatten(), Y.flatten()
    if alpha is None:
        R = np.hypot(X - x.mean(), Y - y.mean())
        R /= R.max()
        R = 1 - R
    else:
        R = [alpha for _ in range(len(X))]
    if color is None:
        C = np.arctan2(Y - y.mean(), X - x.mean())
        C = (C - C.min()) / (C.max() - C.min())
        C = [plt.cm.turbo(c) for c in C]
    else:
        C = [color for _ in range(len(X))]

    fX = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
    if is_push_forward:
        fX, Jf = hxJfx(fX)
    else:
        Jf = f(fX)
    
    fX, Jf = fX.detach().cpu().numpy(), Jf.detach().cpu().numpy()
    X, Y, U, V = fX[..., 0], fX[..., 1], Jf[..., 0], Jf[..., 1]
    lim = min(X.max() - X.min(), Y.max() - Y.min())
    arrow_lim = lim / len(x)
    arrow_scale = arrow_lim / np.max(np.abs(Jf)) * 1.5 * arrow_rescale
    U, V = U * arrow_scale, V * arrow_scale
    arrow_width = arrow_lim / 20
    head_width = arrow_width * 5

    if normlen:
        tmp = np.hypot(U, V)
        U /= (tmp / tmp.max() * 2)
        V /= (tmp / tmp.max() * 2)
    
    # Density plot
    if f_samp_func is not None:
        samples = f_samp_func(n_samples)
        if is_push_forward:
            samples = hxJfx(samples)[0]
        samples = samples.detach().cpu().numpy()
        sns.kdeplot(x=samples[:, 0], y=samples[:, 1], ax=plt.gca(),
                    cmap='gray_r', fill=True, levels=15, cut=10, thresh=0.01,
                    bw_adjust=1)

    for xx, yy, u, v, c, r in zip(X, Y, U, V, C, R):
        plt.arrow(xx, yy, u, v, color=c, width=arrow_width, head_width=head_width, alpha=r)
    
    if plt_traj:
        if X0 is None:
            x0 = np.linspace(x.min(), x.max(), 5)[1:-1]
            y0 = np.linspace(y.min(), y.max(), 5)[1:-1]
            x0, y0 = np.meshgrid(x0, y0, indexing='xy')
            x0, y0 = x0.flatten(), y0.flatten()
            X0 = torch.tensor(np.stack([x0, y0], axis=-1), dtype=torch.float32)
            X0 += torch.randn_like(X0) * X0.std() / 20  # In case of the origin being a fixed point
        traj = torch.full((nSteps, X0.shape[0], 2), float('nan'), device=X0.device)
        with torch.no_grad():
            XT = X0.clone().detach()
            for i in range(nSteps):
                traj[i] = XT
                XT += (f(XT) / step_per_T)
                # Only calculate the trajectory within certain limits
                XT[torch.any(torch.abs(XT) > 100, dim=-1)] = float('nan')
        if is_push_forward:
            traj = hxJfx(traj.reshape(-1, 2))[0].reshape(nSteps, -1, 2)
        plt.plot(traj[:, :, 0].detach().cpu().numpy(), traj[:, :, 1].detach().cpu().numpy(), color=traj_color)
    
    plt.xlim(X.min() - arrow_lim, X.max() + arrow_lim)
    plt.ylim(Y.min() - arrow_lim, Y.max() + arrow_lim)


# %% Simulate trajectories and visualize with PCA

def SimTraj(f: Callable, x0: Tensor, steps: int = 1000, dt: float = 0.1):
    """Simulate trajectories of a dynamical system without noise

    Parameters:
    - f: Callable function representing the dynamical system's vector field
    - x0: Initial conditions of the trajectories, shape (n_trials, n_features)
    - steps: Simulation time steps
    - dt: Time step size

    Returns:
    - traj: Tensor of shape (steps + 1, n_trials, n_features) representing the trajectories
    """
    if len(x0.shape) == 1:
        x0 = x0.unsqueeze(0)
    traj = torch.full((steps + 1, *x0.shape), float('nan'), device=x0.device)
    with torch.no_grad():
        XT = x0.clone().detach()
        traj[0] = XT
        for i in range(steps):
            XT += f(XT) * dt
            # Only calculate the trajectory within certain limits
            XT[torch.any(torch.abs(XT) > 100, dim=-1)] = float('nan')
            traj[i + 1] = XT
    return traj


def PlotTraj(dat: Tensor | ndarray, PCspace: PCA | Tensor | ndarray | None = None,
             plot_dim: int = 3, skip_steps: int = 0, ax=None, samples: Tensor | ndarray | None = None):
    """Plot trajectories and samples in the first 2/3 principal components

    Parameters:
    - dat: Tensor of shape (T, n_trials, n_features)
    - PCspace: PCA object used to transform the data, or another dataset similar to `dat`.
    - plot_dim: Number of dimensions to plot (2 or 3)
    - skip_steps: Number of initial steps to skip in the plot (for clearer visualization)
    - ax: Matplotlib axis to plot on. If None, will create a new figure.
    - samples: Tensor of shape (n_samples, n_features) to plot as a distribution
    
    If PCspace is provided, will plot data in the principal components of PCspace instead.
    If PCspace is None and data has higher dimensionality than plot_dim, will perform PCA
    on the data to reduce to plot_dim dimensions. Otherwise, will plot the data directly.
    """
    
    # Handle input data
    if isinstance(dat, Tensor):
        dat = dat.detach().clone().cpu().numpy()
    if skip_steps > 0:
        dat = dat[skip_steps:]
    T, n = dat.shape[0], dat.shape[2]

    if isinstance(samples, Tensor):
        samples = samples.detach().clone().cpu().numpy()
    
    assert plot_dim <= dat.shape[-1], "Number of dimensions to plot must be less than or equal to the number of features"
    
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        if plot_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

    if PCspace is not None:  # Plot in given PC space
        if isinstance(PCspace, Tensor):
            PCspace = PCspace.detach().clone().cpu().numpy()
        if isinstance(PCspace, ndarray):
            assert PCspace.shape[-1] == n, "PCspace must have the same number of features as dat"
            PCspace = PCA(n_components=plot_dim).fit(PCspace.reshape(-1, n))
        dat_score = PCspace.transform(dat.reshape(-1, n)).reshape(T, -1, PCspace.n_components_)
        if samples is not None:
            samples_score = PCspace.transform(samples)
    
    elif n > plot_dim:  # Perform PCA on data
        PCspace = PCA(n_components=plot_dim).fit(dat.reshape(-1, n))
        dat_score = PCspace.transform(dat.reshape(-1, n)).reshape(T, -1, PCspace.n_components_)
        if samples is not None:
            samples_score = PCspace.transform(samples)
    
    else:  # Plot in original space
        dat_score = dat
        if samples is not None:
            samples_score = samples
    
    dat_score = dat_score.transpose(2, 0, 1)  # Shape (plot_dim, T, n_trials)

    if plot_dim == 3:
        for i in range(dat_score.shape[2]):
            ax.plot(dat_score[0, :, i], dat_score[1, :, i], dat_score[2, :, i], alpha=0.5, color='tab:gray')
        # ax.plot(dat_score[0, 0], dat_score[1, 0], dat_score[2, 0], 'o', color='tab:blue')
        ax.plot(dat_score[0, -1], dat_score[1, -1], dat_score[2, -1], 'x', color='tab:orange')
    else:
        for i in range(dat_score.shape[2]):
            ax.plot(dat_score[0, :, i], dat_score[1, :, i], alpha=0.5, color='tab:gray')
        # ax.plot(dat_score[0, 0], dat_score[1, 0], 'o', color='tab:blue')
        ax.plot(dat_score[0, -1], dat_score[1, -1], 'x', color='tab:orange')
        
        if samples is not None:  # kdeplot
            xl, yl = ax.get_xlim(), ax.get_ylim()
            sns.kdeplot(x=samples_score[:, 0], y=samples_score[:, 1], ax=ax,
                        cmap='YlGn', fill=True, levels=15, cut=10, thresh=0.01,
                        bw_adjust=1)
            ax.set_xlim(xl)
            ax.set_ylim(yl)

    return ax


def PlotDFORM(f: Callable, g: Callable, h: Callable,
              f_samp: Callable | None = None, g_samp: Callable | None = None,
              axes: list | None = None, plot_dim: int = 3, skip_steps: int = 0,
              kwargs_f: dict = {}, kwargs_g: dict = {}, **kwargs):
    """Compare deformed vector fields with target
    
    Parameters:
    - f: Original vector field
    - g: Target vector field
    - h: DFORM model
    - axes: List of three matplotlib axes to plot on. If None, will create a new figure.
    - plot_dim: Number of dimensions to plot (2 or 3)
    - skip_steps: Number of initial steps to skip in the plot (for clearer visualization)
    - kwargs_f/kwargs_g/kwargs: Additional arguments to pass to SimTraj, e.g. steps, dt, etc.
    """
    assert plot_dim <= g.dim, "Number of dimensions to plot must be less than or equal to the number of features"

    if axes is None:
        if plot_dim == 3:
            _, axes = plt.subplots(1, 3, figsize=(12, 6), subplot_kw={'projection': '3d'})
        else:
            _, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    # Sampling function
    if f_samp is None:
        if hasattr(f, 'get_samples'):
            f_samp = f.get_samples
        else:
            f_samp = lambda n: torch.randn(n, f.dim, device=f.device)
    if g_samp is None:
        if hasattr(g, 'get_samples'):
            g_samp = g.get_samples
        else:
            g_samp = lambda n: torch.randn(n, g.dim, device=g.device)

    with torch.no_grad():

        # Example trajectories
        Y0 = torch.randn(60, h.dim, device=h.device)
        X0 = h.inv_deform(Y0)
        trajY = SimTraj(g, Y0[..., :g.dim], **kwargs_g, **kwargs)
        trajX = SimTraj(f, X0, **kwargs_f, **kwargs)
        hTrajX = h.deform(trajX)
        trajY = trajY.detach().cpu()
        trajX = trajX.detach().cpu()
        hTrajX = hTrajX.detach().cpu()

        # Sample distributions
        if plot_dim == 2:
            sample_f = f_samp(1000)
            sample_hf = h.deform(sample_f)
            sample_g = g_samp(1000)
            sample_f = sample_f.detach().cpu()
            sample_hf = sample_hf.detach().cpu()
            sample_g = sample_g.detach().cpu()
        else:
            sample_f, sample_hf, sample_g = None, None, None

        if g.dim > plot_dim:  # Plot in PC space of system g
            input_dim = g.dim
            PCspace = trajY
        else:  # Plot first plot_dim coordinates directly
            input_dim = plot_dim
            PCspace = None
        
        # Plot target vector field
        ax = PlotTraj(trajY[..., :input_dim], PCspace=PCspace, ax=axes[-1],
                      skip_steps=skip_steps, plot_dim=plot_dim, samples=sample_g)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        if plot_dim == 3:
            zl = ax.get_zlim()
        ax.set_title('Target system')
        # Plot original vector field
        ax = PlotTraj(trajX[..., :input_dim], PCspace=PCspace, ax=axes[0],
                      skip_steps=skip_steps, plot_dim=plot_dim, samples=sample_f)
        ax.set_title('Original system')
        # Plot deformed vector field
        ax = PlotTraj(hTrajX[..., :input_dim], PCspace=PCspace, ax=axes[1],
                      skip_steps=skip_steps, plot_dim=plot_dim, samples=sample_hf)
        ax.set_title('Deformed system')
    
    for ax in axes:
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        if g.dim > plot_dim:
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
        else:
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
        if plot_dim == 3:
            ax.set_zlim(zl)
            if g.dim > plot_dim:
                ax.set_zlabel('PC3')
            else:
                ax.set_zlabel('x3')

    return axes


# %% A quick and dirty way to find the stable fixed points
def find_fps(mdl: torch.nn.Module, vel_thres = 1e-5, dist_thres = 1e-2, *args, **kwargs) -> Tensor:
    """Find stable fixed points of the model"""
    n_samples = 300
    with torch.no_grad():
        x0 = torch.randn(n_samples, mdl.dim, device=mdl.device)
        traj = SimTraj(mdl, x0, *args, **kwargs)
        velocity = torch.norm(traj[-10:] - traj[-11:-1], dim=-1)  # (10, n_samples)
        converged = torch.all(velocity < vel_thres, dim=0)  # (n_samples)
        xt = traj[-1][converged]  # (n_converged, dim)
        if xt.shape[0] == 0:
            return torch.empty(0, mdl.dim, device=mdl.device)
        all_fps = [xt[0]]
        for i in range(1, xt.shape[0]):
            dist = [torch.norm(xt[i] - x).item() for x in all_fps]
            if np.min(dist) > dist_thres:
                all_fps.append(xt[i])
        all_fps = torch.stack(all_fps).detach().clone()
    return all_fps


# %% Tests

if __name__ == "__main__":
    from DFORM import *
    dvc = 'cuda' if torch.cuda.is_available() else 'cpu'

    from example_systems import get_RNN, LinearTransformed, get_good_cond_matrix
    n = 32
    f = get_RNN(n, theta_i=[1.5], dvc=dvc)  # Bistable RNN
    H = get_good_cond_matrix(n, ortho=True, dvc=dvc)
    g = LinearTransformed(f, H)
    mdl = Training(f, g, nBatch_linear=3000, nBatch=0, verbose=False)[0]
    f.to('cpu')
    g.to('cpu')
    mdl.to('cpu')
    PlotDFORM(f, g, mdl, dt=0.05, skip_steps=20)
    plt.suptitle(f'Rotated {n}-dimensional RNNs')
    plt.show()

    exit()