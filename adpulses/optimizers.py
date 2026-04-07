from typing import Tuple, Callable, Optional, Union
from time import time
from numbers import Number
from pathlib import Path

import numpy as np
import torch
from torch import optim, Tensor
import mrphy
from mrphy.mobjs import SpinCube, Pulse


def arctanLBFGS(
    target: dict, cube: SpinCube, pulse: Pulse,
    fn_err: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
    fn_pen: Callable[[Tensor], Tensor],
    niter: int = 8, niter_gr: int = 2, niter_rf: int = 2,
    eta: Number = 4.,
    b1Map_: Optional[Tensor] = None, b1Map: Optional[Tensor] = None,
    doQuiet: bool = False, doRelax: bool = True,
    sequence_type: str = 'regular',
    TR: Optional[float] = None,
    vTR: Optional[float] = None,
    alpha: Optional[float] = None,
    alphaDur: Optional[float] = None,
    pulse_save_period: Optional[int] = None,
    pulse_checkpoint_root: Optional[Union[str, Path]] = None,
    excitation_save_period: Optional[int] = None,
    excitation_checkpoint_root: Optional[Union[str, Path]] = None,
) -> Tuple[Pulse, dict]:
    r"""Joint RF/GR optimization via direct arctan trick

    Usage:
        ``arctanLBFGS(target, cube, pulse, fn_err, fn_pen; eta=eta)``

    Inputs:
        - ``target``: dict, with fields:
            ``d_``: `(1, nM, xy)`, desired excitation;
            ``weight_``: `(1, nM)`.
        - ``cube``: mrphy.mobjs.SpinCube.
        - ``pulse``: mrphy.mobjs.Pulse.
        - ``fn_err``: error metric function. See :mod:`~adpulses.metrics`.
        - ``fn_pen``: penalty function. See :mod:`~adpulses.penalties`.
    Optionals:
        - ``niter``: int, number of iterations.
        - ``niter_gr``: int, number of LBFGS iters for updating *gradients*.
        - ``niter_rf``: int, number of LBFGS iters for updating *RF*.
        - ``eta``: `(1,)`, penalization term weighting coefficient.
        - ``b1Map_``: `(1, nM, xy,(nCoils))`, a.u., transmit sensitivity.
        - ``doRelax``: [T/f], whether accounting relaxation effects in simu.
        - ``sequence_type``: str, one of 'regular' (default), 'ss', 'sms';
          case-insensitive.
        - ``TR``: float, repetition time (s). Required for 'ss'/'sms';
          defaults to 55e-3 if None.
        - ``vTR``: float, volume TR (s) for SMS EPI. Defaults to 55e-2 if None.
        - ``alpha``: float, flip angle (deg) for tip-down pulse in 'ss'/'sms'.
          Defaults to 0 if None.
        - ``alphaDur``: float, duration (s) of alpha pulse for 'sms'.
          Defaults to 8e-3 if None.
        - ``pulse_save_period``: int, save pulse every this many outer iters.
        - ``pulse_checkpoint_root``: str or Path, directory for pulse saves.
        - ``excitation_save_period``: int, save Mr_ every this many outer iters.
        - ``excitation_checkpoint_root``: str or Path, directory for Mr_ saves.
    Outputs:
        - ``pulse``: mrphy.mojbs.Pulse, optimized pulse.
        - ``optInfos``: dict, optimization informations.
    """
    sequence_type = sequence_type.lower()
    assert sequence_type in ('regular', 'ss', 'sms'), (
        f"sequence_type must be 'regular', 'ss', or 'sms', got '{sequence_type}'")
    if sequence_type == 'ss':
        if TR is None:
            TR = 55e-3
        if alpha is None:
            alpha = 0.
    elif sequence_type == 'sms':
        if TR is None:
            TR = 55e-3
        if vTR is None:
            vTR = 55e-2
        if alpha is None:
            alpha = 0.
        if alphaDur is None:
            alphaDur = 8e-3

    if pulse_save_period is not None:
        pulse_checkpoint_root = Path(pulse_checkpoint_root)
        pulse_checkpoint_root.mkdir(parents=True, exist_ok=True)
    if excitation_save_period is not None:
        excitation_checkpoint_root = Path(excitation_checkpoint_root)
        excitation_checkpoint_root.mkdir(parents=True, exist_ok=True)
    rfmax, smax = pulse.rfmax, pulse.smax
    eta *= pulse.dt*1e6/4  # normalize eta by dt
    assert ((b1Map_ is None) or (b1Map is None))
    b1Map_ = (b1Map_ if b1Map is None else cube.extract(b1Map))
#    b1Map_ = b1Map_[..., None] if len(b1Map_.shape) == 3 else b1Map_
    # nc = (1 if b1Map_ is None else b1Map_.shape[3])
    # eta /= nc

    # Set up: Interior mapping
    tρ, θ = mrphy.utils.rf2tρθ(pulse.rf, rfmax)
    tsl = mrphy.utils.s2ts(mrphy.utils.g2s(pulse.gr, pulse.dt), smax)

    # enforce contiguousness of optimization variables, o.w. LBFGS may fail
    tρ, θ, tsl = tρ.contiguous(), θ.contiguous(), tsl.contiguous()

    opt_rf = optim.LBFGS([tρ, θ], lr=3., max_iter=10, history_size=30,
                         tolerance_change=1e-4,
                         line_search_fn='strong_wolfe')

    opt_sl = optim.LBFGS([tsl], lr=3., max_iter=40, history_size=60,
                         tolerance_change=1e-6,
                         line_search_fn='strong_wolfe')

    tρ.requires_grad = θ.requires_grad = tsl.requires_grad = True

    # Set up: optimizer
    length = 1+niter*(niter_gr+niter_rf)
    time_hist = np.full((length,), np.nan)
    loss_hist = np.full((length,), np.nan)
    err_hist = np.full((length,), np.nan)
    pen_hist = np.full((length,), np.nan)

    Md_, w_ = target['d_'], target['weight_'].sqrt()  # (1, nM, xy), (1, nM)
    nM = w_.numel()

    def fn_loss(cube, pulse):
        if sequence_type == 'sms':
            Mr_ = cube.applypulse_ss_sms(pulse, b1Map_=b1Map_, doRelax=doRelax,
                                         TR=TR, vTR=vTR, alpha=alpha,
                                         alphaDur=alphaDur)
        elif sequence_type == 'ss':
            Mr_ = cube.applypulse_ss(pulse, b1Map_=b1Map_, doRelax=doRelax,
                                     TR=TR, alpha=alpha)
        else:
            Mr_ = cube.applypulse(pulse, b1Map_=b1Map_, doRelax=doRelax)
        loss_err, loss_pen = fn_err(Mr_, Md_, w_=w_), fn_pen(pulse.rf)
        return loss_err, loss_pen

    log_col = ('\n#iter\t ‖ elapsed time\t ‖ error\t ‖ penalty\t ‖'
               ' total loss\t ‖ avg loss')

    def logger(i, t0, loss, err, pen):
        e, p, lo = err.item(), pen.item(), loss.item()
        msg = (f'{i}\t | {time()-t0:.3f}\t | {e:.3e}\t | {p:.3e}\t | '
               f'{lo:.3e}\t | {lo/nM:.3e}')
        print(msg)
        return loss

    loss_err, loss_pen = fn_loss(cube, pulse)
    loss = loss_err + eta*loss_pen

    logger(0, time(), loss, loss_err, loss_pen)
    time_hist[0], loss_hist[0], err_hist[0], pen_hist[0] = (
        0.0, loss.item(), loss_err.item(), loss_pen.item())

    # Optimization
    t0 = time()
    for i in range(niter):

        if not (i % 5):
            print(log_col)

        log_ind = 0

        def closure():
            opt_rf.zero_grad()
            opt_sl.zero_grad()

            pulse.rf = mrphy.utils.tρθ2rf(tρ, θ, rfmax)
            pulse.gr = mrphy.utils.s2g(mrphy.utils.ts2s(tsl, smax), pulse.dt)

            loss_err, loss_pen = fn_loss(cube, pulse)
            loss = loss_err + eta*loss_pen
            loss.backward()
            return loss

        print('rf-loop: ', niter_rf)
        for _ in range(niter_rf):
            opt_rf.step(closure)

            loss_err, loss_pen = fn_loss(cube, pulse)
            loss = loss_err + eta*loss_pen

            if not doQuiet:
                logger(i+1, t0, loss, loss_err, loss_pen)

            ind = i*(niter_gr+niter_rf)+log_ind+1
            time_hist[ind], loss_hist[ind], err_hist[ind], pen_hist[ind] = (
                time()-t0, loss.item(), loss_err.item(), loss_pen.item())

            log_ind += 1

        print('gr-loop: ', niter_gr)
        for _ in range(niter_gr):
            opt_sl.step(closure)

            loss_err, loss_pen = fn_loss(cube, pulse)
            loss = loss_err + eta*loss_pen

            if not doQuiet:
                logger(i+1, t0, loss, loss_err, loss_pen)

            ind = i*(niter_gr+niter_rf)+log_ind+1
            time_hist[ind], loss_hist[ind], err_hist[ind], pen_hist[ind] = (
                time()-t0, loss.item(), loss_err.item(), loss_pen.item())

            log_ind += 1

        iter_num = i + 1
        if (pulse_save_period is not None
                and iter_num % pulse_save_period == 0):
            fname = pulse_checkpoint_root / f'pulse_iter{iter_num:04d}.pt'
            torch.save(pulse, fname)

        if (excitation_save_period is not None
                and iter_num % excitation_save_period == 0):
            with torch.no_grad():
                Mr_ = cube.applypulse(pulse, b1Map_=b1Map_, doRelax=doRelax)
            fname = (excitation_checkpoint_root
                     / f'excitation_iter{iter_num:04d}.pt')
            torch.save(Mr_.detach().cpu(), fname)

    print('\n== Results: ==')
    print(log_col)
    loss = loss_err + eta*loss_pen

    logger(i+1, t0, loss, loss_err, loss_pen)

    pulse.rf.detach_()
    pulse.gr.detach_()
    optInfos = {'time_hist': time_hist, 'loss_hist': loss_hist,
                'err_hist': err_hist, 'pen_hist': pen_hist}
    return pulse, optInfos
