"""
Main solver
===========

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

The development of this software was sponsored by NAG Ltd. (http://www.nag.co.uk)
and the EPSRC Centre For Doctoral Training in Industrially Focused Mathematical
Modelling (EP/L015803/1) at the University of Oxford. Please contact NAG for
alternative licensing.
"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import warnings

from .diagnostic_info import *
from .exit_information import *
from .model import *
from .params import ParameterList
from .trust_region import trsbox, trsbox_linear
from .util import sumsq, apply_scaling, remove_scaling, eval_objective, \
    random_orthog_directions_new_subspace_within_bounds


__all__ = ['solver']


def min_npt(p):
    return p + 1


def max_npt(p):
    return (p + 1) * (p + 2) // 2


def trust_region_step(gk, Hk, delta):
    # Light trust-region subproblem wrapper (for problems without bound constraints)
    p = len(gk)
    xl = np.full((p,), -1e20)
    xu = np.full((p,), 1e20)
    if Hk is not None:
        sk_red, _, crvmin = trsbox(np.zeros((p,)), gk, Hk, xl, xu, delta)  # step in reduced space
    else:
        sk_red = trsbox_linear(gk, xl, xu, delta)
        crvmin = 0.0
    return sk_red, crvmin


def update_tr(delta, rho, ratio, norm_sk, params):
    if params("tr_radius.update_method") == 'bobyqa':
        if ratio < params("tr_radius.eta1"):  # ratio < 0.1
            iter_type = ITER_ACCEPTABLE if ratio > 0.0 else ITER_UNSUCCESSFUL
            delta = min(params("tr_radius.gamma_dec") * delta, norm_sk)

        elif ratio <= params("tr_radius.eta2"):  # 0.1 <= ratio <= 0.7
            iter_type = ITER_SUCCESSFUL
            delta = max(params("tr_radius.gamma_dec") * delta, norm_sk)

        else:  # (ratio > eta2 = 0.7)
            iter_type = ITER_VERY_SUCCESSFUL
            delta = min(max(params("tr_radius.gamma_inc") * delta, params("tr_radius.gamma_inc_overline") * norm_sk),
                        params("tr_radius.delta_max"))

    else:
        raise RuntimeError("Unknown TR updating method: %s" % params("tr_radius.update_method"))

    if params("tr_radius.use_rho"):
        delta = max(delta, rho)
    return delta, iter_type


def done_with_current_rho(rho, last_successful_iter, current_iter, diffs, crvmin, in_safety=False):
    # crvmin comes from trust region step

    # Wait at least 3 iterations between reductions of rho
    if current_iter <= last_successful_iter + 5:
        return False

    errbig = max(diffs)
    frhosq = 0.125 * rho ** 2
    # if in_safety and crvmin > 0.0 and errbig > frhosq * crvmin:
    #     logging.debug("Not reducing because of this (crvmin = %g)" % crvmin)
    #     return False

    # Otherwise...
    return True


def reduce_rho(delta, rho, rhoend, params):
    if delta > 1.5*rho:  # nothing needed if delta > rho
        return delta, rho

    alpha1 = params("tr_radius.alpha1")
    alpha2 = params("tr_radius.alpha2")
    ratio = rho / rhoend
    if ratio <= 16.0:
        new_rho = rhoend
    elif ratio <= 250.0:
        new_rho = np.sqrt(ratio) * rhoend  # geometric average of rho and rhoend
    else:
        new_rho = alpha1 * rho

    new_delta = max(alpha2 * rho, new_rho)  # self.rho = old rho
    return new_delta, new_rho


def eval_objfun(model, x, objfun, args, scaling_changes, nf, maxfun, params):
    # Simple wrapper to objective evaluation, returning exit codes as needed
    nf += 1
    f = eval_objective(objfun, remove_scaling(x, scaling_changes), args=args,
                          eval_num=nf, full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                          check_for_overflow=params("general.check_objfun_for_overflow"))

    if f <= model.min_objective_value():
        model.save_point(x, f)  # save, since this point was an improvement
        exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")

    elif nf >= maxfun:
        model.save_point(x, f)  # save, just in case this point was an improvement
        exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")

    else:
        exit_info = None

    return f, exit_info, nf


def solve_main(objfun, x0, args, rhobeg, rhoend, maxfun, nruns_so_far, nf_so_far,
               params, scaling_changes, fixed_block_size, hess_npt):
    # First, evaluate objfun at x0 - this gives us m
    logging.debug("Block size %g with %g Hessian points" % (fixed_block_size, hess_npt))
    nf = nf_so_far + 1
    f0 = eval_objective(objfun, remove_scaling(x0, scaling_changes), args=args, eval_num=nf,
                        full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                        check_for_overflow=params("general.check_objfun_for_overflow"))
    exit_info = None

    if f0 <= params("model.abs_tol"):
        exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")

    if exit_info is not None:
        return x0, f0, None, None, nf, nruns_so_far + 1, exit_info, None

    # Initialize model
    delta = rhobeg
    rho = rhobeg if params("tr_radius.use_rho") else 0.0
    model = InterpSet(fixed_block_size, hess_npt, x0, f0, n=len(x0),
                      abs_tol=params("model.abs_tol"), precondition=params("model.precondition_linear_system"))
    exit_info, nf = model.initialise_interp_simplex(delta, objfun, args, scaling_changes, nf, maxfun,
                                                    use_coord_directions=params("geometry.use_coord_directions"),
                                                    box_bound_thresh=params("geometry.direcion_box_bound_thresh"),
                                                    full_x_thresh=params("logging.n_to_print_whole_x_vector"),
                                                    check_for_overflow=params("general.check_objfun_for_overflow"))
    if exit_info is not None:
        xopt, fopt = model.get_final_results()
        return xopt, fopt, None, None, nf, nruns_so_far + 1, exit_info, None

    if params("logging.save_diagnostic_info"):
        diagnostic_info = DiagnosticInfo(x0, f0, delta, rho, nf, with_xk=params("logging.save_xk"),
                                         with_poisedness=params("logging.save_poisedness"),
                                         with_rho=params("tr_radius.use_rho"))
    else:
        diagnostic_info = None

    current_iter = 0
    last_fopts_for_slowterm = []
    num_consecutive_slow_iters = 0
    # Adaptive checking - how many safety/very successful steps have we had in a row so far?
    num_consecutive_safety_steps = 0
    H_to_save, Q_to_save = None, None  # previous model Hessian (for minimal change)
    last_successful_iter = 0  # determines when we can reduce rho
    diffs = [0.0, 0.0, 0.0]

    while True:
        current_iter += 1
        if params("tr_radius.use_rho"):
            logging.debug("*** Iter %g (delta = %g, rho = %g) ***" % (current_iter, delta, rho))
        else:
            logging.debug("*** Iter %g (delta = %g) ***" % (current_iter, delta))
        # print("***** Iter %g *****" % current_iter)

        # Don't bother sampling new points after a safety step - we just reduce delta and try again
        recycle_model = (0 < num_consecutive_safety_steps <= params("geometry.safety_steps_before_redraw"))
        if recycle_model:
            logging.debug("Recycling model from previous iteration (safety step)")

        model.factorise_simplex_system()
        logging.debug("Using model with %g points in p=%g directions" % (model.p+1+model.num_hess_pts, model.p))

        xk = model.xopt()
        fk = model.fopt()
        if params("adaptive.min_change_hess") and H_to_save is not None:
            # Switch previous Hessian to new basis
            # H_to_save is p*p in basis given by Q_to_save, h_old is p*p in basis given by model.Q
            basis_switch = np.dot(Q_to_save.T, model.Q)  # cost O(np^2) to compute, size p*p
            H_old = np.dot(basis_switch.T, np.dot(H_to_save, basis_switch))  # all matrices p*p, cost O(p^3) to compute
            H_old = 0.5*(H_old + H_old.T)  # force symmetry, avoiding rounding errors
        else:
            H_old = None
        interp_ok, ck, gk, Hk = model.build_quadratic_interp_model(Hprev=H_old, model_in_full_space=False)  # based at xopt, in reduced space
        ndirs_used = model.p
        npt_used = model.p + 1 + model.num_hess_pts
        delta_used = delta
        rho_used = rho  # for diagnostic info only
        if not interp_ok:
            exit_info = ExitInformation(EXIT_LINALG_ERROR, "Failed to build interpolation model")
            break  # quit

        # Save Hessian for next iteration (minimal change)
        if params("adaptive.min_change_hess"):
            Q_to_save = model.Q.copy()
            H_to_save = Hk.copy() if Hk is not None else None

        # (optionally) save poisedness of interpolation model
        poisedness = model.poisedness_in_reduced_space(delta) if params("logging.save_poisedness") else None

        # Calculate step/predicted reduction
        sk_red, crvmin = trust_region_step(gk, Hk, delta)  # step in reduced space
        sk_full = model.project_to_full_space(sk_red)
        norm_sk = np.linalg.norm(sk_red)
        xnew = xk + sk_full
        if Hk is not None:
            pred_reduction = -np.dot(sk_red, gk + 0.5 * Hk.dot(sk_red))
        else:
            pred_reduction = -np.dot(sk_red, gk)

        if norm_sk < params("general.safety_step_thresh") * (rho if params("tr_radius.use_rho") else delta) \
                or pred_reduction < 0.0:  # step too short or TRS gave model increase
            logging.debug("Safety step")
            iter_type = ITER_SAFETY
            num_consecutive_safety_steps += 1
            # num_consecutive_very_successful_steps = 0
            ratio = None  # used for diagnostic info only

            if params("tr_radius.use_rho"):
                delta = max(params("tr_radius.gamma_dec") * delta, rho)
                if not done_with_current_rho(rho, last_successful_iter, current_iter, diffs, crvmin,
                                             in_safety=True) or delta > rho:
                    # Delete a bad point (equivalent to fixing geometry)
                    try:
                        sigmas = model.poisedness_of_each_simplex_point(d=sk_red, d_in_full_space=False)
                        sqdists = model.simplex_distances_to_xopt(include_kopt=True)  # ||yt-xk||^2
                        vals = sigmas * np.maximum(sqdists ** 2 / delta ** 4, 1)  # BOBYQA point to remove criterion
                        vals[model.kopt] = -1.0  # make sure kopt is never selected
                        k = np.argmax(vals)
                    except np.linalg.LinAlgError:
                        # If poisedness calculation fails, revert to dropping furthest points
                        sqdists = model.simplex_distances_to_xopt(include_kopt=True)  # ||yt-xk||^2
                        k = np.argmax(sqdists)

                    model.remove_simplex_point(k, check_not_kopt=True)

                    # if params("general.use_safety_step") and norm_sk > rho:
                    #     last_successful_iter = current_iter
                else:
                    delta, rho = reduce_rho(delta, rho, rhoend, params)
                    last_successful_iter = current_iter

                if rho <= rhoend:
                    exit_info = ExitInformation(EXIT_SUCCESS, "rho has reached rhoend")
                    break  # quit
            else:
                delta = params("tr_radius.gamma_dec") * delta
                if delta <= rhoend:
                    exit_info = ExitInformation(EXIT_SUCCESS, "delta has reached rhoend")
                    break  # quit
            # (end of safety step)
        else:
            # (start of normal step)
            num_consecutive_safety_steps = 0

            # Evaluate objective at xnew
            fnew, exit_info, nf = eval_objfun(model, xnew, objfun, args, scaling_changes, nf, maxfun, params)
            if exit_info is not None:
                break  # quit

            # Slow termination checking
            if fnew < fk:
                if len(last_fopts_for_slowterm) <= params("slow.history_for_slow"):
                    last_fopts_for_slowterm.append(fk)
                    num_consecutive_slow_iters = 0
                else:
                    last_fopts_for_slowterm = last_fopts_for_slowterm[1:] + [fk]
                    this_iter_slow = (np.log(last_fopts_for_slowterm[0]) - np.log(fk)) / float(
                        params("slow.history_for_slow")) < params("slow.thresh_for_slow")
                    if this_iter_slow:
                        num_consecutive_slow_iters += 1
                    else:
                        num_consecutive_slow_iters = 0

                # Do slow termination check
                if num_consecutive_slow_iters >= params("slow.max_slow_iters"):
                    model.save_point(xnew, fnew)  # save, since this point was an improvement
                    exit_info = ExitInformation(EXIT_SLOW_WARNING, "Maximum slow iterations reached")
                    break  # quit

            # Decide on type of step
            actual_reduction = fk - fnew
            ratio = actual_reduction / pred_reduction
            if min(norm_sk, delta) > rho:  # if ||sk|| >= rho, successful!
                last_successful_iter = current_iter
            diffs = [abs(actual_reduction - pred_reduction), diffs[0], diffs[1]]

            # Update trust region radius
            delta, iter_type = update_tr(delta, rho, ratio, norm_sk, params)

            logging.debug("Ratio = %g (%s)" % (ratio, iter_type))

            # Add xnew to interpolation set
            if model.p < model.n:
                logging.debug("Appending xk+sk")
                model.append_simplex_point(xnew, fnew)  # updates xopt
                xnew_appended = True
            else:
                # If the model is full, replace xnew with the point furthest from xk (from previous iteration)
                try:
                    sigmas = model.poisedness_of_each_simplex_point(d=sk_red, d_in_full_space=False)
                    sqdists = model.simplex_distances_to_xopt(include_kopt=True)  # ||yt-xk||^2
                    vals = sigmas * np.maximum(sqdists ** 2 / delta ** 4, 1)  # BOBYQA point to remove criterion
                    vals[model.kopt] = -1.0  # make sure kopt is never selected
                    knew = np.argmax(vals)
                except np.linalg.LinAlgError:
                    # If poisedness calculation fails, revert to dropping furthest points
                    sqdists = model.simplex_distances_to_xopt(include_kopt=True)  # ||yt-xk||^2
                    knew = np.argmax(sqdists)
                logging.debug("Changing simplex point %g to xk+sk" % knew)
                model.change_simplex_point(knew, xnew, fnew, check_not_kopt=True)  # updates xopt
                xnew_appended = False

            # Drop points no longer needed
            # Remove at least 1 from xnew (if appended) and 1 to make space for a new direction in next iter
            min_npt_to_drop = (2 if xnew_appended else 1)
            p_drop = (1.0 - params("adaptive.simplex_save_frac_successful")) * model.p
            if iter_type in [ITER_ACCEPTABLE, ITER_UNSUCCESSFUL]:
                p_drop = (1.0 - params("adaptive.simplex_save_frac_unsuccessful")) * model.p
                if params("hessian.forget_on_unsuccessful"):
                    H_to_save = None
                if params("hessian.drop_interp_points_on_unsuccessful"):
                    model.clear_hessian_points()
            ndirs_to_keep = max(0, min(model.p - int(p_drop), model.p - min_npt_to_drop))  # TODO this is correct
            # ndirs_to_keep = max(0, min(int(p_drop), model.p - min_npt_to_drop))
            ndirs_to_drop = model.p - ndirs_to_keep
            logging.debug("After adding xnew, model has %g directions, dropping %g" % (model.p, ndirs_to_drop))

            # Criteria of points to remove:
            try:
                sigmas = model.poisedness_of_each_simplex_point(delta=delta)
                sqdists = model.simplex_distances_to_xopt(include_kopt=True)  # ||yt-xk||^2
                vals = sigmas * np.maximum(sqdists ** 2 / delta ** 4, 1)  # BOBYQA point to remove criterion
            except np.linalg.LinAlgError:
                # If poisedness calculation fails, revert to dropping furthest points
                logging.debug("Poisedness failed, using distance only")
                vals = model.simplex_distances_to_xopt(include_kopt=True)  # ||yt-xk||^2
            vals[model.kopt] = -np.inf  # make sure kopt is never selected
            # logging.debug("kopt = %g, argmin = %g" % (model.kopt, np.argmin(model.fvals)))
            # logging.debug("Here, fvals in [%.20g, %.20g]" % (np.min(model.fvals), np.max(model.fvals)))

            for i in range(ndirs_to_drop):
                k = np.argmax(vals)
                # logging.debug("Dropping %g (val = %g) // kopt = %g with val = %g" % (k, vals[k], model.kopt, vals[model.kopt]))
                vals = np.delete(vals, k)  # keep vals indices in line with indices of model.points
                model.remove_simplex_point(k, check_not_kopt=True)
                # try:
                #     model.remove_simplex_point(k, check_not_kopt=True)
                #     # logging.debug("Model kopt = %g, with val = %g" % (model.kopt, vals[model.kopt]))
                # except AssertionError as e:
                #     for i in range(model.p + 1):
                #         logging.debug("%g: %g" % (i, vals[i]))
                #     raise e

            if params("tr_radius.use_rho"):
                if ratio < 0 and done_with_current_rho(rho, last_successful_iter, current_iter, diffs, crvmin,
                                                       in_safety=False) and delta <= rho:
                    delta, rho = reduce_rho(delta, rho, rhoend, params)
                    last_successful_iter = current_iter

                if rho <= rhoend:
                    exit_info = ExitInformation(EXIT_SUCCESS, "rho has reached rhoend")
                    break  # quit
            else:
                if delta <= rhoend:
                    exit_info = ExitInformation(EXIT_SUCCESS, "delta has reached rhoend")
                    break  # quit
            # (end of normal step)

        # Add new simplex points to get back to correct size - do in both safety and normal steps
        ndirs_to_add = fixed_block_size - model.p
        logging.debug("Model has p=%g, adding %g points" % (model.p, ndirs_to_add))
        for i in range(ndirs_to_add):
            model.factorise_simplex_system()  # now can get model.Q (None if no directions stored currently, which is ok for next line)
            d = random_orthog_directions_new_subspace_within_bounds(1, delta, model.xl - model.xopt(),
                                                                    model.xu - model.xopt(),
                                                                    Q=model.Q,
                                                                    box_bound_thresh=params(
                                                                        "geometry.direcion_box_bound_thresh"))[0, :]

            # Evaluate objective
            xnew = model.xopt() + d
            fnew, exit_info, nf = eval_objfun(model, xnew, objfun, args, scaling_changes, nf, maxfun, params)
            if exit_info is not None:
                break  # quit (inner loop only)

            # Append xnew to model
            model.append_simplex_point(xnew, fnew)
        if exit_info is not None:
            break  # quit main loop
        logging.debug("At end of iteration, now have %g directions" % (model.p))

        # (in both safety and normal steps, update diagnostic info)
        if params("logging.save_diagnostic_info"):
            diagnostic_info.save_info(model, delta_used, rho_used, Hk, ndirs_used, npt_used, norm_sk, np.linalg.norm(gk),
                                      nruns_so_far + 1, nf, current_iter, iter_type, ratio, poisedness)
        continue

    xopt, fopt = model.get_final_results()
    return xopt, fopt, None, None, nf, nruns_so_far + 1, exit_info, diagnostic_info  # TODO no gradient/Hessian returned?


def solve(objfun, x0, args=(), fixed_block_size=None, hess_npt=None, bounds=None, rhobeg=None, rhoend=1e-8, maxfun=None,
          user_params=None, objfun_has_noise=False, scaling_within_bounds=False):
    n = len(x0)

    if fixed_block_size is not None:
        assert 1 <= fixed_block_size <= n, "fixed_block_size, if specified, must be in [1..n]"
    else:
        fixed_block_size = n  # default to full block

    if hess_npt is not None:
        assert min_npt(fixed_block_size) <= fixed_block_size + 1 + hess_npt <= max_npt(fixed_block_size), \
            "Invalid choice of hess_npt %g (%g points outside [%g,%g] for block size %g)" \
            % (hess_npt, fixed_block_size + 1 + hess_npt, min_npt(fixed_block_size), max_npt(fixed_block_size), fixed_block_size)
    else:
        hess_npt = fixed_block_size  # default to 2p+1 interpolation points in total

    # Set missing inputs (if not specified) to some sensible defaults
    if bounds is None:
        xl = None
        xu = None
        scaling_within_bounds = False
    else:
        raise RuntimeError(
            'Block Py-BOBYQA does not support bounds yet (needs update to trust region subproblem solver)')
        # assert len(bounds) == 2, "bounds must be a 2-tuple of (lower, upper), where both are arrays of size(x0)"
        # xl = bounds[0]
        # xu = bounds[1]

    if xl is None:
        xl = np.full((n,), -1e20)  # unconstrained
    if xu is None:
        xu = np.full((n,), 1e20)  # unconstrained

    if rhobeg is None:
        rhobeg = 0.1 if scaling_within_bounds else 0.1 * max(np.max(np.abs(x0)), 1.0)
    if maxfun is None:
        maxfun = min(100 * (n + 1), 1000)  # 100 gradients, capped at 1000

    # Set parameters
    params = ParameterList(int(n), int(maxfun), objfun_has_noise=objfun_has_noise)  # make sure int, not np.int
    if user_params is not None:
        for (key, val) in user_params.items():
            params(key, new_value=val)

    scaling_changes = None
    if scaling_within_bounds:
        shift = xl.copy()
        scale = xu - xl
        scaling_changes = (shift, scale)

    x0 = apply_scaling(x0, scaling_changes)
    xl = apply_scaling(xl, scaling_changes)
    xu = apply_scaling(xu, scaling_changes)

    exit_info = None
    # Input & parameter checks
    if exit_info is None and rhobeg < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhobeg must be strictly positive")

    if exit_info is None and rhoend < 0.0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhoend must be strictly positive")

    if exit_info is None and rhobeg <= rhoend:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "rhobeg must be > rhoend")

    if exit_info is None and maxfun <= 0:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "maxfun must be strictly positive")

    if exit_info is None and np.shape(x0) != (n,):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "x0 must be a vector")

    if exit_info is None and np.shape(x0) != np.shape(xl):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "lower bounds must have same shape as x0")

    if exit_info is None and np.shape(x0) != np.shape(xu):
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "upper bounds must have same shape as x0")

    if exit_info is None and np.min(xu - xl) < 2.0 * rhobeg:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "gap between lower and upper must be at least 2*rhobeg")

    if maxfun <= n:
        warnings.warn("maxfun <= n: Are you sure your budget is large enough?", RuntimeWarning)

    # Check invalid parameter values

    all_ok, bad_keys = params.check_all_params()
    if exit_info is None and not all_ok:
        exit_info = ExitInformation(EXIT_INPUT_ERROR, "Bad parameters: %s" % str(bad_keys))

    # If we had an input error, quit gracefully
    if exit_info is not None:
        exit_flag = exit_info.flag
        exit_msg = exit_info.message(with_stem=True)
        results = OptimResults(None, None, None, None, 0, 0, exit_flag, exit_msg)
        return results

    # Enforce lower & upper bounds on x0
    idx = (xl < x0) & (x0 <= xl + rhobeg)
    if np.any(idx):
        warnings.warn("x0 too close to lower bound, adjusting", RuntimeWarning)
    x0[idx] = xl[idx] + rhobeg

    idx = (x0 <= xl)
    if np.any(idx):
        warnings.warn("x0 below lower bound, adjusting", RuntimeWarning)
    x0[idx] = xl[idx]

    idx = (xu - rhobeg <= x0) & (x0 < xu)
    if np.any(idx):
        warnings.warn("x0 too close to upper bound, adjusting", RuntimeWarning)
    x0[idx] = xu[idx] - rhobeg

    idx = (x0 >= xu)
    if np.any(idx):
        warnings.warn("x0 above upper bound, adjusting", RuntimeWarning)
    x0[idx] = xu[idx]

    # Call main solver (first time)
    nruns = 0
    nf = 0
    xmin, fmin, gradmin, hessmin, nf, nruns, exit_info, diagnostic_info = \
        solve_main(objfun, x0, args, rhobeg, rhoend, maxfun, nruns, nf, params, scaling_changes,
                   fixed_block_size, hess_npt)

    # Process final return values & package up
    exit_flag = exit_info.flag
    exit_msg = exit_info.message(with_stem=True)

    # Un-scale gradient & Hessian
    if scaling_changes is not None:
        if gradmin is not None:
            gradmin = gradmin / scaling_changes[1]
        if hessmin is not None:
            hessmin = hessmin / np.outer(scaling_changes[1], scaling_changes[1])

    results = OptimResults(remove_scaling(xmin, scaling_changes), fmin, gradmin, hessmin, nf, nruns, exit_flag,
                           exit_msg)

    if params("logging.save_diagnostic_info") and diagnostic_info is not None:
        df = diagnostic_info.to_dataframe()
        results.diagnostic_info = df

    return results
