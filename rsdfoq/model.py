"""
Model
====

Maintain a class which represents an interpolating set.
This class should calculate the various geometric quantities of interest to us.


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

from math import sqrt
import numpy as np
import scipy.linalg as linalg

from .exit_information import *
# from .hessian import Hessian, to_upper_triangular_vector
from .trust_region import trsbox
from .util import random_orthog_directions_new_subspace_within_bounds, eval_objective, remove_scaling, model_value

__all__ = ['InterpSet', 'project_to_full_space', 'project_to_reduced_space']


class InterpSet(object):
    def __init__(self, subspace_dim, max_hess_npt, x0, f0, n=None, abs_tol=-1e20, precondition=True):
        if n is None:
            n = len(x0)
        assert 1 <= subspace_dim <= n, "Subspace dimension must be in [1..n], got %g" % subspace_dim
        assert 0 <= max_hess_npt <= n*(n+1)//2, "Max number of pts for Hessian must be in [0..n(n+1)/2], got %g" % max_hess_npt
        assert x0.shape == (n,), "x0 has wrong shape (got %s, expect (%g,))" % (str(x0.shape), n)
        # assert xl.shape == (n,), "xl has wrong shape (got %s, expect (%g,))" % (str(xl.shape), n)
        # assert xu.shape == (n,), "xu has wrong shape (got %s, expect (%g,))" % (str(xu.shape), n)
        self.n = n
        self.p = subspace_dim
        self.max_hess_npt = max_hess_npt
        self.have_hess = (self.max_hess_npt > 0)
        self.num_hess_pts = 0  # how many points have we got in the Hessian component so far?

        # Bounds (not used right now)
        self.xl = np.full((self.n,), -1e20, dtype=float)
        self.xu = np.full((self.n,), 1e20, dtype=float)

        # Interpolation points
        # All points are in the form (self.xbase + direction)
        self.points = np.full((self.p+1, self.n), np.nan, dtype=float)  # main set of interpolation directions (for gradient)
        self.points2 = None  # when add points, will be secondary set of Hessian interp points (size self.num_hess_pts, self.n)
        self.points[0,:] = x0

        # Function values
        self.fvals = np.full((self.p+1,), np.inf, dtype=float)  # objective value for each row of self.points
        self.fvals2 = None  # objective value for each row of self.points2
        self.fvals[0] = f0

        self.kopt = 0

        # Termination criteria
        self.abs_tol = abs_tol

        # Saved point - always check this value before quitting solver
        self.xsave = x0.copy()
        self.fsave = f0

        # Factorisation information
        self.simplex_factorisation_current = False
        self.Q = None  # basis for subspace, size n*p
        self.R = None  # self.points - self.xopt() directions come from Q*R, size p*p
        self.R2 = None  # self.points2 - self.xopt() directions come from Q*R2, size p*num_hess_pts
        self.precondition = precondition  # should the interpolation matrix be preconditioned?

    def xopt(self):
        return self.xpt(self.kopt)

    def fopt(self):
        return self.fval(self.kopt)

    def xpt(self, k):
        assert 0 <= k <= self.p + self.num_hess_pts, "Invalid index %g" % k
        if k <= self.p:
            return self.points[k, :]
        else:
            return self.points2[k - self.p - 1, :]

    def fval(self, k):
        assert 0 <= k <= self.p + self.num_hess_pts, "Invalid index %g" % k
        if k <= self.p:
            return self.fvals[k]
        else:
            return self.fvals2[k - self.p - 1]

    def min_objective_value(self):
        # Get termination criterion for f small: f <= abs_tol or f <= rel_tol * f0
        return self.abs_tol

    def save_point(self, x, f):
        if self.fsave is None or f <= self.fsave:
            self.xsave = x.copy()
            self.fsave = f
            return True
        else:
            return False  # this value is worse than what we have already - didn't save

    def get_final_results(self):
        # Return x and fval for optimal point (either from xsave+fsave or kopt)
        if self.fsave is None or self.fopt() <= self.fsave:  # optimal has changed since xsave+fsave were last set
            return self.xopt(), self.fopt()
        else:
            return self.xsave, self.fsave

    def initialise_interp_simplex(self, delta, objfun, args, scaling_changes, nf, maxfun, use_coord_directions=False,
                                  box_bound_thresh=0.01, full_x_thresh=6, check_for_overflow=True):
        assert delta > 0.0, "delta must be strictly positive"

        # Called upon initialisation only
        x0 = self.xpt(0)

        # Get directions
        dirns = np.zeros((self.p, self.n))  # each row is an offset from x0
        if use_coord_directions:
            idx_choices = np.random.choice(self.n, size=self.p, replace=False)
            for i in range(self.p):
                idx = idx_choices[i]
                # Decide whether to do +delta or -delta (usually via a coin toss)
                upper_gap = self.xu[idx] - x0[idx]
                lower_gap = x0[idx] - self.xl[idx]
                if min(lower_gap, upper_gap) <= box_bound_thresh * delta:
                    # If very close to boundary on at least one side, just go in the larger direction we can step
                    step = -min(delta, lower_gap) if lower_gap > upper_gap else min(delta, upper_gap)
                else:
                    step = min(delta, upper_gap) if np.random.random() >= 0.5 else -min(delta, lower_gap)

                dirns[i, idx] = step
        else:
            dirns = random_orthog_directions_new_subspace_within_bounds(self.p, delta, self.xl - x0, self.xu - x0,
                                                                        Q=None,
                                                                        box_bound_thresh=box_bound_thresh)

        # Evaluate objective at these points
        exit_info = None
        for i in range(self.p):
            x = x0 + dirns[i, :]  # point to evaluate

            if nf >= maxfun:
                exit_info = ExitInformation(EXIT_MAXFUN_WARNING, "Objective has been called MAXFUN times")
                break  # quit

            nf += 1
            f = eval_objective(objfun, remove_scaling(x, scaling_changes), args=args, eval_num=nf,
                               full_x_thresh=full_x_thresh, check_for_overflow=check_for_overflow)

            if f < self.min_objective_value():
                self.save_point(x, f)
                exit_info = ExitInformation(EXIT_SUCCESS, "Objective is sufficiently small")
                break  # quit

            self.points[i + 1, :] = x
            self.fvals[i + 1] = f

        # Choose kopt as best value so far
        self.kopt = np.argmin(self.fvals)

        self.simplex_factorisation_current = False
        return exit_info, nf

    def clear_hessian_points(self):
        # Remove all Hessian points
        if self.have_hess:
            self.points2 = None
            self.fvals2 = None
            self.num_hess_pts = 0
        return

    def append_simplex_point(self, x, f):
        assert self.p < self.n, "Cannot append points to full-dimensional interpolation set"
        self.points = np.append(self.points, x.reshape((1, self.n)), axis=0)  # append row to xpt
        self.fvals = np.append(self.fvals, f)  # append entry to fvals
        self.p += 1

        if f < self.fopt():
            self.kopt = self.p

        self.simplex_factorisation_current = False
        return

    def append_hessian_point(self, x, f):
        assert self.have_hess, "Cannot append Hessian point when using linear models"

        if self.num_hess_pts >= self.max_hess_npt:
            # Need to drop an existing point
            k_to_drop = 0  # drop oldest Hessian point
            self.points2 = np.delete(self.points2, k_to_drop, axis=0)
            self.fvals2 = np.delete(self.fvals2, k_to_drop)
            self.num_hess_pts -= 1

        if self.num_hess_pts == 0:
            self.points2 = x.reshape((1,self.n)).copy()
            self.fvals2 = np.array([f])
        else:
            self.points2 = np.append(self.points2, x.reshape((1,self.n)), axis=0)  # append row to end
            self.fvals2 = np.append(self.fvals2, f)
        self.num_hess_pts += 1
        return

    def change_simplex_point(self, k, x, f, check_not_kopt=True, move_to_hessian=True):
        # Update point k to x (w.r.t. xbase), with residual values fvec
        # If move_to_hessian, put point in self.points2
        assert 0 <= k <= self.p, "Invalid index %g" % k
        if check_not_kopt:
            assert k != self.kopt, "Cannot remove current iterate from interpolation set"
        else:
            self.save_point(self.xopt(), self.fopt())  # if we are going to delete kopt, make sure we save it first

        if move_to_hessian and self.have_hess:
            self.append_hessian_point(self.points[k, :], self.fvals[k])

        self.points[k, :] = x
        self.fvals[k] = f
        self.simplex_factorisation_current = False

        self.kopt = np.argmin(self.fvals)
        return

    def remove_simplex_point(self, k, check_not_kopt=True, move_to_hessian=True):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        assert self.p >= 1, "Need to keep at least one point (iterate) in interpolation set"
        if check_not_kopt:
            assert k != self.kopt, "Cannot remove current iterate from interpolation set"

        if move_to_hessian and self.have_hess:
            self.append_hessian_point(self.points[k, :], self.fvals[k])

        self.points = np.delete(self.points, k, axis=0)  # delete row
        self.fvals = np.delete(self.fvals, k)
        self.p -= 1

        # Make sure kopt is always the best value we have in the simplex
        # Need to update even if k!=kopt (e.g. deleted point before kopt)
        self.kopt = np.argmin(self.fvals)
        self.simplex_factorisation_current = False
        return

    def remove_hessian_points(self, npts_to_drop):
        actual_npts_to_drop = max(min(npts_to_drop, self.num_hess_pts), 0)
        if actual_npts_to_drop > 0:
            # Drop oldest points (the top rows of self.points2)
            self.points2 = np.delete(self.points2, np.arange(actual_npts_to_drop), axis=0)
            self.fvals2 = np.delete(self.fvals2, np.arange(actual_npts_to_drop))
            self.num_hess_pts -= actual_npts_to_drop
        return

    def simplex_directions_from_xopt(self):
        dirns = self.points - self.xopt()  # subtract from each row of matrix (numpy does automatically)
        return np.delete(dirns, self.kopt, axis=0)  # drop kopt-th entry / kopt-th row

    def simplex_distances_to_xopt(self, include_kopt=True):
        dirns = self.points - self.xopt()  # subtract from each row of matrix (numpy does automatically)
        distances = np.linalg.norm(dirns, axis=1)  # norm of each row
        if include_kopt:
            return distances
        else:
            return np.delete(distances, self.kopt)  # drop kopt-th entry

    def factorise_simplex_system(self):
        if not self.simplex_factorisation_current:
            if self.p > 0:
                dirns = self.simplex_directions_from_xopt()  # size (p, n)
                self.Q, self.R = linalg.qr(dirns.T, mode='economic', pivoting=False)  # Q is n*p, R is p*p
            else:
                self.Q, self.R = None, None
            self.simplex_factorisation_current = True
        return

    def construct_generic_simplex_model(self, vals_to_interpolate, gradient_in_full_space=False):
        self.factorise_simplex_system()  # ensure factorisation up-to-date
        c = vals_to_interpolate[self.kopt]
        rhs = np.delete(vals_to_interpolate - c, self.kopt)  # drop kopt-th entry
        g = linalg.solve_triangular(self.R, rhs, trans='T')  # R.T \ rhs -> gradient in reduced space

        # model based at xopt
        if gradient_in_full_space:
            return c, self.project_to_full_space(g)
        else:
            return c, g

    def build_simplex_interp_model(self, gradient_in_full_space=False):
        try:
            c, g = self.construct_generic_simplex_model(self.fvals, gradient_in_full_space=gradient_in_full_space)
        except:
            return False, None, None  # flag error

        if not vector_ok(c) and vector_ok(g):
            return False, None, None  # flag error

        return True, c, g  # model based at xopt

    def simplex_lagrange_poly(self, k, gradient_in_full_space=False):
        assert 0 <= k <= self.p, "Invalid index %g" % k
        vals_to_interpolate = np.zeros((self.p + 1,))
        vals_to_interpolate[k] = 1.0  # rest are zero
        try:
            c, g = self.construct_generic_simplex_model(vals_to_interpolate, gradient_in_full_space=gradient_in_full_space)
        except:
            return False, None, None  # flag error

        if not (vector_ok(c) and vector_ok(g)):
            return False, None, None  # flag error

        return True, c, g  # model based at xopt

    def poisedness_of_each_simplex_point(self, delta=None, d=None, d_in_full_space=False):
        # Return the poisedness of each point in the interpolation set
        # if delta is set, then calculate the maximum of |L(xopt+s)| over all ||s||<=delta for each point
        # if d is set, calculate |L(xopt+d)| for each point
        assert (delta is not None or d is not None) and (delta is None or d is None), "Must specify exactly one of delta and d"
        poisedness = np.zeros((self.p + 1,))
        for k in range(self.p + 1):
            interp_ok, c, g = self.simplex_lagrange_poly(k, gradient_in_full_space=d_in_full_space)
            if not interp_ok:
                poisedness[k] = np.nan
            elif delta is not None:
                normg = np.linalg.norm(g)
                # Maximiser/minimiser of (linear) Lagrange poly in B(xopt, delta) is s = +/- delta/||g|| * g
                # Value is c +/- delta*||g||
                poisedness[k] = max(abs(c + delta * normg), abs(c - delta * normg))
            else:
                poisedness[k] = abs(c + np.dot(g, d))
        return poisedness

    def simplex_poisedness(self, delta):
        # Calculate poisedness constant in ball around xopt (everything in reduced space)
        return np.max(self.poisedness_of_each_simplex_point(delta=delta))

    def construct_generic_quadratic_model(self, simplex_vals_to_interpolate, hessian_vals_to_interpolate, Hprev=None,
                                          model_in_full_space=False):
        self.factorise_simplex_system()  # ensure factorisation up-to-date

        if self.have_hess and self.num_hess_pts > 0:
            # Build full interpolation model with c, g, and H all together
            # If use simplex to get c and g, then hessian points to only build H, then will lose interpolation at
            # simplex points

            # Merge both sets of points
            simplex_dirns_from_xopt = self.points - self.xopt()  # subtract from each row of matrix (numpy does automatically)
            simplex_dirns_from_xopt = np.delete(simplex_dirns_from_xopt, self.kopt, axis=0)  # kopt-th row
            hess_dirns_from_xopt = self.points2[:self.num_hess_pts, :] - self.xopt()
            Y_full = np.concatenate((simplex_dirns_from_xopt, hess_dirns_from_xopt), axis=0).T  # yt-xk on each column
            vals_to_interpolate = np.concatenate((simplex_vals_to_interpolate, hessian_vals_to_interpolate[:self.num_hess_pts]))
            npt = len(vals_to_interpolate)

            # Build interpolation matrix (in reduced space)
            Y_red = self.Q.T.dot(Y_full)  # (yt-xk) in reduced coords on each column
            # Note: rows of Y_red for Hessian points are approx (yt-xk not in col Q)
            distances_to_xopt = np.linalg.norm(Y_full, axis=1)
            approx_delta = sqrt(np.max(distances_to_xopt)) if self.precondition else 1.0  # largest distance to xopt ~ delta
            A, left_scaling, right_scaling = build_interpolation_matrix(Y_red, approx_delta=approx_delta)

            # Solve linear system
            c = vals_to_interpolate[self.kopt]
            if npt == self.p + 1 or npt == (self.p + 1) * (self.p + 2) // 2:
                rhs = np.delete(vals_to_interpolate - c, self.kopt)  # drop kopt-th entry
            else:
                rhs = np.zeros((npt + self.p - 1,))
                rhs[:npt - 1] = np.delete(vals_to_interpolate - c, self.kopt)
                if Hprev is not None:
                    # Modified to be minimum *change* in Hessian, rather than minimum Hessian
                    for t in range(npt - 1):
                        dx = Y_red[:, t]
                        rhs[t] = rhs[t] - 0.5 * np.dot(dx, Hprev.dot(dx))  # include old Hessian
            coeffs = np.linalg.solve(A, rhs * left_scaling) * right_scaling  # solve linear system

            # Construct model (note: c defined above)
            if npt == self.p + 1:
                g = coeffs.copy()
                H = None
            elif npt == (self.p + 1) * (self.p + 2) // 2:
                g = coeffs[:self.p]
                # H = Hessian(self.p, coeffs[self.p:])  # rest of coeffs are upper triangular part of Hess
                H = build_symmetric_matrix_from_vector(self.p, coeffs[self.p:])
            else:
                g = coeffs[npt - 1:]  # last n values
                H = np.zeros((self.p, self.p)) if Hprev is None else Hprev  # min change Hessian?
                for i in range(npt - 1):
                    dx = Y_red[:, i]
                    H += coeffs[i] * np.outer(dx, dx)

        else:
            # Only building linear models - easy case
            c, g = self.construct_generic_simplex_model(simplex_vals_to_interpolate, gradient_in_full_space=False)
            H = None

        if model_in_full_space:
            cfull = c
            gfull = self.project_to_full_space(g)
            Hfull = np.dot(self.Q, np.dot(H, self.Q.T)) if H is not None else None
            return cfull, gfull, Hfull  # based at self.xopt()
        else:
            return c, g, H  # based at self.xopt()

    def build_quadratic_interp_model(self, Hprev=None, model_in_full_space=False):
        # Note: Hprev needs to be p*p (in *current* self.Q basis)
        try:
            c, g, H = self.construct_generic_quadratic_model(self.fvals, self.fvals2, Hprev=Hprev,
                                                             model_in_full_space=model_in_full_space)
        except:
            return False, None, None, None  # flag error

        if self.have_hess and self.num_hess_pts > 0:
            if not (vector_ok(c) and vector_ok(g) and vector_ok(H)):
                return False, None, None, None  # flag error
        else:  # don't check Hessian
            if not (vector_ok(c) and vector_ok(g) and H is None):
                return False, None, None, None  # flag error

        return True, c, g, H  # model based at xopt

    def lagrange_poly(self, k, model_in_full_space=False):
        assert 0 <= k < self.p + 1 + self.num_hess_pts, "Invalid k (got %g, must be in [0,%g])" % (k, self.p+1+self.num_hess_pts)
        simplex_vals_to_interpolate = np.zeros((self.p+1,))
        hessian_vals_to_interpolate = np.zeros((self.num_hess_pts))
        if k < self.p+1:
            simplex_vals_to_interpolate[k] = 1.0
        else:
            hessian_vals_to_interpolate[k-self.p-1] = 1.0
        return self.construct_generic_quadratic_model(simplex_vals_to_interpolate, hessian_vals_to_interpolate,
                                                      Hprev=None, model_in_full_space=model_in_full_space)

    def poisedness_in_reduced_space(self, delta):
        # Calculate poisedness constant in ball around xopt (everything in reduced space)
        lmax = None
        for k in range(self.p + 1 + self.num_hess_pts):
            c, g, H = self.lagrange_poly(k, model_in_full_space=False)
            if H is not None:
                # Quadratic models
                dmin, _, _ = trsbox(self.project_to_reduced_space(self.xopt()), g, H, -1e20*np.ones((self.p,)), 1e20*np.ones((self.p,)), delta)
                dmax, _, _ = trsbox(self.project_to_reduced_space(self.xopt()), -g, -H, -1e20 * np.ones((self.p,)), 1e20 * np.ones((self.p,)), delta)
                l = max(abs(c + model_value(g, H, dmin)), abs(c + model_value(g, H, dmax)))
            else:
                # Linear models
                dmin = -delta * g / np.linalg.norm(g)
                dmax = delta * g / np.linalg.norm(g)
                l = max(abs(c + np.dot(g, dmin)), abs(c + np.dot(g, dmax)))
            if lmax is None or l > lmax:
                lmax = l
        return lmax

    def project_to_full_space(self, x):
        assert self.simplex_factorisation_current, "Cannot project, factorisation is invalid"
        return project_to_full_space(self.Q, x)

    def project_to_reduced_space(self, x):
        assert self.simplex_factorisation_current, "Cannot project, factorisation is invalid"
        return project_to_reduced_space(self.Q, x)


def project_to_full_space(Q, x):
    return Q.dot(x)


def project_to_reduced_space(Q, x):
    return Q.T.dot(x)


def vector_ok(x):
    return np.all(np.isfinite(x))


def build_interpolation_matrix(Y, approx_delta=1.0):
    # Routine from Py-BOBYQA
    # Y has columns Y[:,t] = yt - xk
    n, p = Y.shape  # p = npt-1
    assert n + 1 <= p + 1 <= (n + 1) * (n + 2) // 2, "npt must be in range [n+1, (n+1)(n+2)/2]"

    # What scaling was applied to each part of the matrix?
    # A(scaled) = diag(left_scaling) * A(unscaled) * diag(right_scaling)

    if p == n:  # linear models
        A = Y.T / approx_delta
        left_scaling = np.ones((n,))  # no left scaling
        right_scaling = np.ones((n,)) / approx_delta
    elif p + 1 == (n + 1) * (n + 2) // 2:  # fully quadratic models
        A = np.zeros((p, p))
        A[:, :n] = Y.T / approx_delta
        for i in range(p):
            A[i, n:] = get_entry_vector_from_symmetric_matrix(np.outer(Y[:, i], Y[:, i]) - 0.5 * np.diag(np.square(Y[:, i]))) / (approx_delta ** 2)
            # A[i, n:] = to_upper_triangular_vector(np.outer(Y[:, i], Y[:, i]) - 0.5 * np.diag(np.square(Y[:, i]))) / (approx_delta ** 2)
        left_scaling = np.ones((p,))  # no left scaling
        right_scaling = np.ones((p,))
        right_scaling[:n] = 1.0 / approx_delta
        right_scaling[n:] = 1.0 / (approx_delta ** 2)
    else:  # underdetermined quadratic models
        A = np.zeros((p + n, p + n))
        for i in range(p):
            for j in range(p):
                A[i, j] = 0.5 * np.dot(Y[:, i], Y[:, j]) ** 2 / (approx_delta ** 4)
        A[:p, p:] = Y.T / approx_delta
        A[p:, :p] = Y / approx_delta
        left_scaling = np.ones((p + n,))
        right_scaling = np.ones((p + n,))
        left_scaling[:p] = 1.0 / (approx_delta ** 2)
        left_scaling[p:] = approx_delta
        right_scaling[:p] = 1.0 / (approx_delta ** 2)
        right_scaling[p:] = approx_delta
    return A, left_scaling, right_scaling


def build_symmetric_matrix_from_vector(n, entries):
    assert entries.shape == (n*(n+1)//2,), "Entries vector has wrong size, got %g, expect %g (for n=%g)" % (len(entries), n*(n+1)//2, n)
    A = np.zeros((n, n))
    ih = -1
    for j in range(n):  # j = 0, ..., n-1
        for i in range(j + 1):  # i = 0, ..., j
            ih += 1
            A[i, j] = entries[ih]
            A[j, i] = entries[ih]
    return A


def get_entry_vector_from_symmetric_matrix(A):
    n = A.shape[0]
    hq = np.zeros((n*(n+1)//2,))
    ih = -1
    for j in range(n):  # j = 0, ..., n-1
        for i in range(j + 1):  # i = 0, ..., j
            ih += 1
            hq[ih] = A[i,j]
    return hq