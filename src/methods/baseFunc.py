import abc
from torch import nn
import torch
from . import misc

def linear_interp(t0, y0, t1, y1, t):
    assert t0 <= t <= t1, f"Incorrect time order for linear interpolation: t0={t0}, t={t}, t1={t1}."
    y = (t1 - t) / (t1 - t0) * y0 + (t - t0) / (t1 - t0) * y1
    return y

class BaseSDE(abc.ABC, nn.Module):
    """Base class for all SDEs.

    Inheriting from this class ensures `noise_type` and `sde_type` are valid attributes, which the solver depends on.
    """

    def __init__(self, noise_type, sde_type):
        super(BaseSDE, self).__init__()
        # Making these Python properties breaks `torch.jit.script`.
        self.noise_type = noise_type
        self.sde_type = sde_type

class RenameMethodsSDE(BaseSDE):

    def __init__(self, sde, drift='f', diffusion='g', prior_drift='h', diffusion_prod='g_prod',
                 drift_and_diffusion='f_and_g', drift_and_diffusion_prod='f_and_g_prod'):
        super(RenameMethodsSDE, self).__init__(noise_type=sde.noise_type, sde_type=sde.sde_type)
        self._base_sde = sde
        for name, value in zip(('f', 'g', 'h', 'g_prod', 'f_and_g', 'f_and_g_prod'),
                               (drift, diffusion, prior_drift, diffusion_prod, drift_and_diffusion,
                                drift_and_diffusion_prod)):
            try:
                setattr(self, name, getattr(sde, value))
            except AttributeError:
                pass

class ForwardSDE(BaseSDE):

    def __init__(self, sde, fast_dg_ga_jvp_column_sum=False):
        super(ForwardSDE, self).__init__(sde_type=sde.sde_type, noise_type=sde.noise_type)
        self._base_sde = sde
        self.f_and_g_prod = self.f_and_g_prod_default2

        self.f = getattr(sde, 'f', self.f_default)
        self.g = getattr(sde, 'g', self.g_default)
        self.f_and_g = getattr(sde, 'f_and_g', self.f_and_g_default)
        self.g_prod = getattr(sde, 'g_prod', self.g_prod_default)
        self.prod = {
            'diagonal': self.prod_diagonal
        }.get(sde.noise_type, self.prod_default)

        self.g_prod_and_gdg_prod = {
            'diagonal': self.g_prod_and_gdg_prod_diagonal,
            'additive': self.g_prod_and_gdg_prod_additive,
        }.get(sde.noise_type, self.g_prod_and_gdg_prod_default)
        self.dg_ga_jvp_column_sum = {
            'general': (
                self.dg_ga_jvp_column_sum_v2 if fast_dg_ga_jvp_column_sum else self.dg_ga_jvp_column_sum_v1
            )
        }.get(sde.noise_type, self._return_zero)

    ########################################
    #                  f                   #
    ########################################
    def f_default(self, t, y):
        raise RuntimeError("Method `f` has not been provided, but is required for this method.")

    ########################################
    #                  g                   #
    ########################################
    def g_default(self, t, y):
        raise RuntimeError("Method `g` has not been provided, but is required for this method.")

    ########################################
    #               f_and_g                #
    ########################################

    def f_and_g_default(self, t, y):
        return self.f(t, y), self.g(t, y)

    ########################################
    #                prod                  #
    ########################################

    def prod_diagonal(self, g, v):
        return g * v

    def prod_default(self, g, v):
        return misc.batch_mvp(g, v)

    ########################################
    #                g_prod                #
    ########################################

    def g_prod_default(self, t, y, v):
        return self.prod(self.g(t, y), v)

    ########################################
    #             f_and_g_prod             #
    ########################################

    def f_and_g_prod_default1(self, t, y, v):
        return self.f(t, y), self.g_prod(t, y, v)

    def f_and_g_prod_default2(self, t, y, v):
        with torch.enable_grad():
            f, g = self.f_and_g(t, y)
        return f, self.prod(g, v)

    ########################################
    #          g_prod_and_gdg_prod         #
    ########################################

    # Computes: g_prod and sum_{j, l} g_{j, l} d g_{j, l} d x_i v2_l.
    def g_prod_and_gdg_prod_default(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2.unsqueeze(-2),
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_diagonal(self, t, y, v1, v2):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            vg_dg_vjp, = misc.vjp(
                outputs=g,
                inputs=y,
                grad_outputs=g * v2,
                retain_graph=True,
                create_graph=requires_grad,
                allow_unused=True
            )
        return self.prod(g, v1), vg_dg_vjp

    def g_prod_and_gdg_prod_additive(self, t, y, v1, v2):
        return self.g_prod(t, y, v1), 0.

    ########################################
    #              dg_ga_jvp               #
    ########################################

    # Computes: sum_{j,k,l} d g_{i,l} / d x_j g_{j,k} A_{k,l}.
    def dg_ga_jvp_column_sum_v1(self, t, y, a):
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)
            dg_ga_jvp = [
                misc.jvp(
                    outputs=g[..., col_idx],
                    inputs=y,
                    grad_inputs=ga[..., col_idx],
                    retain_graph=True,
                    create_graph=requires_grad,
                    allow_unused=True
                )[0]
                for col_idx in range(g.size(-1))
            ]
            dg_ga_jvp = sum(dg_ga_jvp)
        return dg_ga_jvp

    def dg_ga_jvp_column_sum_v2(self, t, y, a):
        # Faster, but more memory intensive.
        requires_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            y = y if y.requires_grad else y.detach().requires_grad_(True)
            g = self.g(t, y)
            ga = torch.bmm(g, a)

            batch_size, d, m = g.size()
            y_dup = torch.repeat_interleave(y, repeats=m, dim=0)
            g_dup = self.g(t, y_dup)
            ga_flat = ga.transpose(1, 2).flatten(0, 1)
            dg_ga_jvp, = misc.jvp(
                outputs=g_dup,
                inputs=y_dup,
                grad_inputs=ga_flat,
                create_graph=requires_grad,
                allow_unused=True
            )
            dg_ga_jvp = dg_ga_jvp.reshape(batch_size, m, d, m).permute(0, 2, 1, 3)
            dg_ga_jvp = dg_ga_jvp.diagonal(dim1=-2, dim2=-1).sum(-1)
        return dg_ga_jvp

    def _return_zero(self, t, y, v):  # noqa
        return 0.


class BaseBrownian(metaclass=abc.ABCMeta):
    __slots__ = ()

    @abc.abstractmethod
    def __call__(self, ta, tb=None, return_U=False, return_A=False):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shape(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def levy_area_approximation(self):
        raise NotImplementedError

    def size(self):
        return self.shape

