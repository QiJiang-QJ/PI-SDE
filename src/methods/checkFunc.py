import warnings

import torch

from .baseFunc import RenameMethodsSDE, ForwardSDE
from . import misc
from ._brownian import BrownianInterval


def check_contract(sde, y0, ts, bm, method, adaptive, options, names, logqp):
    if names is None:
        names_to_change = {}
    else:
        names_to_change = {key: names[key] for key in ("drift", "diffusion", "prior_drift", "drift_and_diffusion",
                                                       "drift_and_diffusion_prod") if key in names}
    if len(names_to_change) > 0:
        sde = RenameMethodsSDE(sde, **names_to_change)

    if not torch.is_tensor(y0):
        raise ValueError("`y0` must be a torch.Tensor.")
    if y0.dim() != 2:
        raise ValueError("`y0` must be a 2-dimensional tensor of shape (batch, channels).")

    if not torch.is_tensor(ts):
        if not isinstance(ts, (tuple, list)) or not all(isinstance(t, (float, int)) for t in ts):
            raise ValueError("Evaluation times `ts` must be a 1-D Tensor or list/tuple of floats.")
        ts = torch.tensor(ts, dtype=y0.dtype, device=y0.device)
    if not misc.is_strictly_increasing(ts):
        raise ValueError("Evaluation times `ts` must be strictly increasing.")

    batch_sizes = []
    state_sizes = []
    noise_sizes = []
    batch_sizes.append(y0.size(0))
    state_sizes.append(y0.size(1))
    if bm is not None:
        if len(bm.shape) != 2:
            raise ValueError("`bm` must be of shape (batch, noise_channels).")
        batch_sizes.append(bm.shape[0])
        noise_sizes.append(bm.shape[1])

    def _check_2d(name, shape):
        if len(shape) != 2:
            raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
        batch_sizes.append(shape[0])
        state_sizes.append(shape[1])

    def _check_2d_or_3d(name, shape):
        if sde.noise_type == 'diagonal':
            if len(shape) != 2:
                raise ValueError(f"{name} must be of shape (batch, state_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[1])
        else:
            if len(shape) != 3:
                raise ValueError(f"{name} must be of shape (batch, state_channels, noise_channels), but got {shape}.")
            batch_sizes.append(shape[0])
            state_sizes.append(shape[1])
            noise_sizes.append(shape[2])

    has_f = False
    has_g = False
    if hasattr(sde, 'f'):
        has_f = True
        f_drift_shape = tuple(sde.f(ts[0], y0).size())
        _check_2d('Drift', f_drift_shape)
    if hasattr(sde, 'g'):
        has_g = True
        g_diffusion_shape = tuple(sde.g(ts[0], y0).size())
        _check_2d_or_3d('Diffusion', g_diffusion_shape)
    if hasattr(sde, 'f_and_g'):
        has_f = True
        has_g = True
        _f, _g = sde.f_and_g(ts[0], y0)
        f_drift_shape = tuple(_f.size())
        g_diffusion_shape = tuple(_g.size())
        _check_2d('Drift', f_drift_shape)
        _check_2d_or_3d('Diffusion', g_diffusion_shape)

    for batch_size in batch_sizes[1:]:
        if batch_size != batch_sizes[0]:
            raise ValueError("Batch sizes not consistent.")
    for state_size in state_sizes[1:]:
        if state_size != state_sizes[0]:
            raise ValueError("State sizes not consistent.")
    for noise_size in noise_sizes[1:]:
        if noise_size != noise_sizes[0]:
            raise ValueError("Noise sizes not consistent.")

    if sde.noise_type == 'scalar':
        if noise_sizes[0] != 1:
            raise ValueError(f"Scalar noise must have only one channel; the diffusion has {noise_sizes[0]} noise "
                             f"channels.")

    sde = ForwardSDE(sde)

    if bm is None:
        bm = BrownianInterval(t0=ts[0], t1=ts[-1], size=(batch_sizes[0], noise_sizes[0]), dtype=y0.dtype,
                              device=y0.device, levy_area_approximation='none')

    if options is None:
        options = {}
    else:
        options = options.copy()

    return sde, y0, ts, bm, method, options