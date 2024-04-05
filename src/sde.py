import torch
from torch import nn
import warnings

from methods import checkFunc,Euler,misc
from adjoint_sde import AdjointSDE
from methods._brownian import BaseBrownian, ReverseBrownian
from methods.types import Any, Dict, Optional, Scalar, Tensor, Tensors, TensorOrTensors, Vector

class _SdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sde, ts, dt, bm, solver, method, adjoint_method, adjoint_adaptive, adjoint_rtol, adjoint_atol,
                dt_min, adjoint_options, len_extras, y0, *extras_and_adjoint_params):
        ctx.sde = sde
        ctx.dt = dt
        ctx.bm = bm
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_adaptive = adjoint_adaptive
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.dt_min = dt_min
        ctx.adjoint_options = adjoint_options
        ctx.len_extras = len_extras

        extra_solver_state = extras_and_adjoint_params[:len_extras]
        adjoint_params = extras_and_adjoint_params[len_extras:]

        # This .detach() is VERY IMPORTANT. See adjoint_sde.py::AdjointSDE.get_state.
        y0 = y0.detach()
        # Necessary for the same reason
        extra_solver_state = tuple(x.detach() for x in extra_solver_state)
        ys, extra_solver_state = solver.integrate(y0, ts, extra_solver_state)

        # Else just remove the `extra_solver_state` information.
        ctx.saved_extras_for_backward = False
        extras_for_backward = ()
        ctx.save_for_backward(ys, ts, *extras_for_backward, *adjoint_params)
        return (ys, *extra_solver_state)

    @staticmethod
    def backward(ctx, grad_ys, *grad_extra_solver_state):  # noqa
        ys, ts, *extras_and_adjoint_params = ctx.saved_tensors
        if ctx.saved_extras_for_backward:
            extra_solver_state = extras_and_adjoint_params[:ctx.len_extras]
            adjoint_params = extras_and_adjoint_params[ctx.len_extras:]
        else:
            grad_extra_solver_state = ()
            extra_solver_state = None
            adjoint_params = extras_and_adjoint_params

        aug_state = [ys[-1], grad_ys[-1]] + list(grad_extra_solver_state) + [torch.zeros_like(param)
                                                                             for param in adjoint_params]
        shapes = [t.size() for t in aug_state]
        aug_state = misc.flatten(aug_state)
        aug_state = aug_state.unsqueeze(0)  # dummy batch dimension
        adjoint_sde = AdjointSDE(ctx.sde, adjoint_params, shapes)
        reverse_bm = ReverseBrownian(ctx.bm)

        solver = Euler.Euler(
            sde=adjoint_sde,
            bm=reverse_bm,
            dt=ctx.dt,
            adaptive=ctx.adjoint_adaptive,
            rtol=ctx.adjoint_rtol,
            atol=ctx.adjoint_atol,
            dt_min=ctx.dt_min,
            options=ctx.adjoint_options
        )
        if extra_solver_state is None:
            extra_solver_state = solver.init_extra_solver_state(ts[-1], aug_state)

        for i in range(ys.size(0) - 1, 0, -1):
            (_, aug_state), *extra_solver_state = _SdeintAdjointMethod.apply(adjoint_sde,
                                                                             torch.stack([-ts[i], -ts[i - 1]]),
                                                                             ctx.dt,
                                                                             reverse_bm,
                                                                             solver,
                                                                             ctx.adjoint_method,
                                                                             ctx.adjoint_method,
                                                                             ctx.adjoint_adaptive,
                                                                             ctx.adjoint_rtol,
                                                                             ctx.adjoint_atol,
                                                                             ctx.dt_min,
                                                                             ctx.adjoint_options,
                                                                             len(extra_solver_state),
                                                                             aug_state,
                                                                             *extra_solver_state,
                                                                             *adjoint_params)
            aug_state = misc.flat_to_shape(aug_state.squeeze(0), shapes)
            aug_state[0] = ys[i - 1]
            aug_state[1] = aug_state[1] + grad_ys[i - 1]
            if i != 1:
                aug_state = misc.flatten(aug_state)
                aug_state = aug_state.unsqueeze(0)  # dummy batch dimension

        if ctx.saved_extras_for_backward:
            out = aug_state[1:]
        else:
            out = [aug_state[1]] + ([None] * ctx.len_extras) + aug_state[2:]
        return (
            None, None, None, None, None, None, None, None, None, None, None, None, None, *out,
        )


def sdeint_adjoint(sde: nn.Module,
                   y0: Tensor,
                   ts: Vector,
                   bm: Optional[BaseBrownian] = None,
                   method: Optional[str] = None,
                   adjoint_method: Optional[str] = None,
                   dt: Scalar = 1e-3,
                   adaptive: bool = False,
                   adjoint_adaptive: bool = False,
                   rtol: Scalar = 1e-5,
                   adjoint_rtol: Scalar = 1e-5,
                   atol: Scalar = 1e-4,
                   adjoint_atol: Scalar = 1e-4,
                   dt_min: Scalar = 1e-5,
                   options: Optional[Dict[str, Any]] = None,
                   adjoint_options: Optional[Dict[str, Any]] = None,
                   adjoint_params=None,
                   names: Optional[Dict[str, str]] = None,
                   logqp: bool = False,
                   extra: bool = False,
                   extra_solver_state: Optional[Tensors] = None,
                   **unused_kwargs) -> TensorOrTensors:

    misc.handle_unused_kwargs(unused_kwargs, msg="`sdeint_adjoint`")
    del unused_kwargs

    if adjoint_params is None and not isinstance(sde, nn.Module):
        raise ValueError('`sde` must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    sde, y0, ts, bm, method, options = checkFunc.check_contract(sde, y0, ts, bm, method, adaptive, options, names, logqp)
    misc.assert_no_grad(['ts', 'dt', 'rtol', 'adjoint_rtol', 'atol', 'adjoint_atol', 'dt_min'],
                        [ts, dt, rtol, adjoint_rtol, atol, adjoint_atol, dt_min])
    adjoint_params = tuple(sde.parameters()) if adjoint_params is None else tuple(adjoint_params)
    adjoint_params = filter(lambda x: x.requires_grad, adjoint_params)
    adjoint_options = {} if adjoint_options is None else adjoint_options.copy()

    solver = Euler.Euler(
        sde=sde,
        bm=bm,
        dt=dt,
        adaptive=adaptive,
        rtol=rtol,
        atol=atol,
        dt_min=dt_min,
        options=options
    )
    if extra_solver_state is None:
        extra_solver_state = solver.init_extra_solver_state(ts[0], y0)

    ys, *extra_solver_state = _SdeintAdjointMethod.apply(
        sde, ts, dt, bm, solver, method, adjoint_method, adjoint_adaptive, adjoint_rtol, adjoint_atol, dt_min,
        adjoint_options, len(extra_solver_state), y0, *extra_solver_state, *adjoint_params
    )

    return ys

