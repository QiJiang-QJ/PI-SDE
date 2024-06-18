import abc
import torch
from .baseFunc import BaseSDE,BaseBrownian,linear_interp
from .types import Scalar, Tensor, Dict, Tensors, Tuple

class ABCMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super(ABCMeta, cls).__call__(*args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), '__is_abstract_attribute__', False)
        }
        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with"
                " abstract attributes: {}".format(
                    cls.__name__,
                    ', '.join(abstract_attributes)
                )
            )
        return instance

class BaseSDESolver(metaclass=ABCMeta):
    """API for solvers with possibly adaptive time stepping."""
    def __init__(self,
                 sde: BaseSDE,
                 bm: BaseBrownian,
                 dt: Scalar,
                 adaptive: bool,
                 rtol: Scalar,
                 atol: Scalar,
                 dt_min: Scalar,
                 options: Dict,
                 **kwargs):
        super(BaseSDESolver, self).__init__(**kwargs)
        self.sde = sde
        self.bm = bm
        self.dt = dt
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.options = options

    def init_extra_solver_state(self, t0, y0) -> Tensors:
        return ()
    @abc.abstractmethod
    def step(self, t0: Scalar, t1: Scalar, y0: Tensor, extra0: Tensors) -> Tuple[Tensor, Tensors]:

        raise NotImplementedError
    def integrate(self, y0: Tensor, ts: Tensor, extra0: Tensors) -> Tuple[Tensor, Tensors]:
        step_size = self.dt
        prev_t = curr_t = ts[0]
        prev_y = curr_y = y0
        curr_extra = extra0
        ys = [y0]
        for out_t in ts[1:]:
            while curr_t < out_t:
                next_t = min(curr_t + step_size, ts[-1])
                prev_t, prev_y = curr_t, curr_y
                curr_y, curr_extra = self.step(curr_t, next_t, curr_y, curr_extra)
                curr_t = next_t
            ys.append(linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

        return torch.stack(ys, dim=0), curr_extra





class Euler(BaseSDESolver):
    def __init__(self,sde,**kwargs):
        super(Euler, self).__init__(sde=sde,**kwargs)
        self.strong_order = 1.0 if sde.noise_type == 'additive' else 0.5

    def step(self, t0, t1, y0, extra0):
        del extra0
        dt = t1 - t0
        I_k = self.bm(t0, t1)

        f, g_prod = self.sde.f_and_g_prod(t0, y0, I_k)
        y1 = y0 + f * dt + g_prod
        return y1, ()
