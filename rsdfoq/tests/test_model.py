"""

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

import numpy as np
import unittest

from rsdfoq.model import *
from rsdfoq.util import model_value


def overall_qr_error(A, Q, R):
    # Check various properties of QR factorisation
    m, n = A.shape
    if m < n:
        raise RuntimeError("overall_qr_error not designed for m < n case (A.shape = %s)" % A.shape)
    factorisation_error = np.linalg.norm(A - Q.dot(R))
    if factorisation_error > 1e-10:
        print("- Factorisation error = %g" % factorisation_error)
    Q_orthog_error = np.linalg.norm(np.eye(n) - Q.T.dot(Q))
    if Q_orthog_error > 1e-10:
        print("- Q orthogonal error = %g" % Q_orthog_error)
    R_triang_error = 0.0
    if R_triang_error > 1e-10:
        print("- R upper triangular error = %g" % R_triang_error)
    if R.shape != (n,n):
        print(" - R has wrong shape %s, expect (%g,%g)" % (R.shape, n, n))
    return max(factorisation_error, Q_orthog_error, R_triang_error)


def rosenbrock(x):
    # x0 = np.array([-1.2, 1.0])
    r = np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
    return np.dot(r, r)


def powell_singular(x):
    # x0 = np.array([3.0, -1.0, 0.0, 1.0])
    fvec = np.zeros((4,))

    fvec[0] = x[0] + 10.0 * x[1]
    fvec[1] = np.sqrt(5.0) * (x[2] - x[3])
    fvec[2] = (x[1] - 2.0 * x[2]) ** 2
    fvec[3] = np.sqrt(10.0) * (x[0] - x[3]) ** 2

    return np.dot(fvec, fvec)


class NumpyTestCase(unittest.TestCase):
    def assertArrayEqual(self, first, second, rtol=1e-06, atol=1e-08, msg=""):
        np.testing.assert_allclose(first, second, rtol=rtol, atol=atol, err_msg=msg)


class InterpSetTestCase(NumpyTestCase):
    def generic_model_check(self, model, npt, simplex_pts, hess_pts, objfun, kopt, mystr="", rtol=1e-6, atol=1e-8):
        xopt = simplex_pts[kopt]
        p = len(simplex_pts) - 1  # p+1 points

        # Basic check: is the interpolation set as expected?
        self.assertEqual(model.p, p, msg="Wrong p %s" % mystr)
        for k in range(model.p + 1):
            xk = simplex_pts[k]
            self.assertArrayEqual(model.xpt(k), xk, rtol=rtol, atol=atol, msg="Simplex point %g wrong %s" % (k, mystr))
            self.assertAlmostEqual(model.fval(k), objfun(xk), msg="Simplex fval %g wrong %s" % (k, mystr))
        if hess_pts is not None:
            self.assertEqual(model.num_hess_pts, len(hess_pts), msg="Wrong num Hessian pts %s" % mystr)
            for k in range(model.num_hess_pts):
                xk = hess_pts[k]
                self.assertArrayEqual(model.xpt(k+p+1), xk, rtol=rtol, atol=atol, msg="Hess point %g wrong %s" % (k, mystr))
                self.assertAlmostEqual(model.fval(k+p+1), objfun(xk), msg="Hess fval %g wrong %s" % (k, mystr))
        else:
            self.assertEqual(model.num_hess_pts, 0, msg="Wrong num Hessian pts (should be zero) %s" % mystr)

        # Make sure we have the correct current optimum
        self.assertEqual(model.kopt, kopt, msg="Wrong kopt %s" % mystr)
        self.assertArrayEqual(model.xopt(), xopt, rtol=rtol, atol=atol, msg="Wrong xopt %s" % mystr)
        self.assertAlmostEqual(model.fopt(), objfun(xopt), msg="Wrong fopt %s" % mystr)

        # Basic functionality testing
        dirs = model.simplex_directions_from_xopt()
        dists_with_kopt = model.simplex_distances_to_xopt(include_kopt=True)
        dists_without_kopt = model.simplex_distances_to_xopt(include_kopt=False)
        for i in range(model.p+1):
            s = simplex_pts[i]-simplex_pts[kopt]
            if i == kopt:
                self.assertAlmostEqual(np.linalg.norm(s), dists_with_kopt[i], msg="Wrong dist 1 for pt %g %s" % (i, mystr))
                continue  # skipt
            j = i if i < kopt else i-1
            self.assertAlmostEqual(np.linalg.norm(s), dists_with_kopt[i], msg="Wrong dist 1 for pt %g %s" % (i, mystr))
            self.assertAlmostEqual(np.linalg.norm(s), dists_without_kopt[j], msg="Wrong dist 2 for pt %g %s" % (i, mystr))
            self.assertArrayEqual(s, dirs[j,:], msg="Wrong dirn %g %s" % (i, mystr))

        # Check simplex interpolation model
        is_ok, c, g = model.build_simplex_interp_model(gradient_in_full_space=False)
        is_ok2, c2, g2 = model.build_simplex_interp_model(gradient_in_full_space=True)
        self.assertTrue(is_ok, msg="Simplex interp failed %s" % mystr)
        self.assertTrue(is_ok2, msg="Simplex interp 2 failed %s" % mystr)
        for k in range(model.p + 1):
            xk = model.xpt(k)
            sk_full = xk - model.xopt()
            sk = model.project_to_reduced_space(sk_full)
            self.assertAlmostEqual(c + np.dot(g, sk), objfun(xk), msg="Simplex interp wrong for k=%g %s" % (k, mystr))
            self.assertAlmostEqual(c2 + np.dot(g2, sk_full), objfun(xk), msg="Simplex interp 2 wrong for k=%g %s" % (k, mystr))

        # Check simplex Lagrange polynomials
        for k in range(model.p + 1):
            is_ok, c, g = model.simplex_lagrange_poly(k, gradient_in_full_space=False)
            is_ok2, c2, g2 = model.simplex_lagrange_poly(k, gradient_in_full_space=True)
            self.assertTrue(is_ok, msg="Simplex Lagrange %g interp failed %s" % (k, mystr))
            self.assertTrue(is_ok2, msg="Simplex Lagrange %g interp 2 failed %s" % (k, mystr))
            for j in range(model.p + 1):
                expected_value_at_xj = 1.0 if j==k else 0.0
                sj_full = model.xpt(j) - model.xopt()
                sj = model.project_to_reduced_space(sj_full)
                self.assertAlmostEqual(c + np.dot(g, sj), expected_value_at_xj,
                                       msg="Lagrange %g value wrong at x%g %s" % (k, j, mystr))
                self.assertAlmostEqual(c2 + np.dot(g2, sj_full), expected_value_at_xj,
                                       msg="Lagrange %g value 2 wrong at x%g %s" % (k, j, mystr))

        # Full quadratic interpolation model (only works when have a full Hessian ready)
        if model.num_hess_pts >= npt:
            self.assertEqual(model.num_hess_pts, model.hess_npt, "Wrong num Hess points %s" % mystr)
            self.assertEqual(model.num_hess_pts, npt, "Wrong num Hess points 2 %s" % mystr)

            # Interpolate with hess_old = None (i.e. zeros)
            is_ok, c, g, H = model.interpolate_model(model_in_full_space=False, hess_old=None)
            is_ok2, c2, g2, H2 = model.interpolate_model(model_in_full_space=True, hess_old=None)
            self.assertTrue(is_ok, msg="Full interp failed %s" % mystr)
            self.assertTrue(is_ok2, msg="Full interp 2 failed %s" % mystr)
            for k in range(model.p + 1 + model.num_hess_pts):
                xk = model.xpt(k)
                sk_full = xk - model.xopt()
                sk = model.project_to_reduced_space(sk_full)
                self.assertAlmostEqual((c + np.dot(g, sk)) if H is None else (c + model_value(g, H, sk)), objfun(xk),
                                       msg="Full interp wrong for k=%g %s" % (k, mystr))
                self.assertAlmostEqual(
                    (c2 + np.dot(g2, sk_full)) if H2 is None else (c2 + model_value(g2, H2, sk_full)), objfun(xk),
                    msg="Full interp 2 wrong for k=%g %s" % (k, mystr))

                # Interpolate with hess_old = Identity
                hess_old = np.eye(model.p, model.p)
                is_ok3, c3, g3, H3 = model.interpolate_model(model_in_full_space=False, hess_old=hess_old)
                is_ok4, c4, g4, H4 = model.interpolate_model(model_in_full_space=True, hess_old=hess_old)
                self.assertTrue(is_ok3, msg="Full interp 3 failed %s" % mystr)
                self.assertTrue(is_ok4, msg="Full interp 4 failed %s" % mystr)
                for k in range(model.p + 1 + model.num_hess_pts):
                    xk = model.xpt(k)
                    sk_full = xk - model.xopt()
                    sk = model.project_to_reduced_space(sk_full)
                    self.assertAlmostEqual((c3 + np.dot(g3, sk)) if H3 is None else (c3 + model_value(g3, H3, sk)),
                                           objfun(xk),
                                           msg="Full interp 3 wrong for k=%g %s" % (k, mystr))
                    self.assertAlmostEqual(
                        (c4 + np.dot(g4, sk_full)) if H4 is None else (c4 + model_value(g4, H4, sk_full)), objfun(xk),
                        msg="Full interp 4 wrong for k=%g %s" % (k, mystr))

                # Check Hessians are closer to the expected value (in reduced space, because it's easier)
                self.assertLessEqual(np.linalg.norm(H, ord='fro'), np.linalg.norm(H3, ord='fro'),
                                     msg="Full interp Hess 1 distance wrong %s" % mystr)
                self.assertLessEqual(np.linalg.norm(H3-hess_old, ord='fro'),
                                     np.linalg.norm(H-hess_old, ord='fro'),
                                     msg="Full interp Hess 2 distance wrong %s" % mystr)
        return


class TestInterpFullDim_2np1(InterpSetTestCase):
    def runTest(self):
        # Full-dimensional model for Rosenbrock
        objfun = rosenbrock
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n = 2
        npt = 2*n + 1
        x0 = np.array([-1.2, 1.0])
        f0 = objfun(x0)
        model = InterpSet(n, npt-n-1, x0, f0, abs_tol=-1e20, precondition=True)
        self.assertEqual(model.n, n, msg="Wrong n")
        self.assertEqual(model.p, n, msg="Wrong p")
        self.assertEqual(model.max_hess_npt, n, msg="Wrong Hessian npt")
        self.assertTrue(model.have_hess, msg="Model has Hessian")
        self.assertEqual(model.num_hess_pts, 0, msg="Wrong num_hess_pts")
        self.assertFalse(model.simplex_factorisation_current, msg="Factorisation not up-to-date")
        self.assertArrayEqual(model.xpt(0), x0, msg="Wrong x0")
        self.assertArrayEqual(model.fval(0), f0, msg="Wrong f0")
        np.random.seed(0)
        delta = 1.0
        nf = 1
        maxfun = 10
        exit_info, nf = model.initialise_interp_simplex(delta, objfun, args, scaling_changes, nf, maxfun)
        # Extracted manually, given np.random.seed(0)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
        # for i in range(1, n+1):  # points[0] is x0
        #     print("Point %g" % i, model.points[i,:] - x0)
        x1 = x0 + delta * np.array([-0.874428832365212, -0.485153807702683])
        x2 = x0 + delta * np.array([-0.485153807702683, 0.874428832365212])
        self.assertAlmostEqual(np.linalg.norm(x1 - x0), delta, msg="distance x1-x0 not delta")
        self.assertAlmostEqual(np.linalg.norm(x2 - x0), delta, msg="distance x2-x0 not delta")
        self.assertAlmostEqual(np.dot(x1 - x0, x2 - x0), 0.0, msg="x1 and x2 not orthogonal")
        self.assertFalse(model.simplex_factorisation_current, msg="Wrong factorisation_current after init")
        self.generic_model_check(model, npt, [x0, x1, x2], [], objfun, 0, mystr="After init")
        self.assertIsNone(exit_info, msg="Init failed")
        self.assertEqual(nf, n+1, msg="Wrong nf after init")

        # Append Hessian point (not an improvement)
        x3 = np.array([5.0, 5.0])
        model.append_hessian_point(x3, objfun(x3))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3], objfun, 0, mystr="After adding x3")

        # Append Hessian point (which is an improvement) - this is also enough to test interpolation
        x4 = np.array([1.0, 1.0])
        model.append_hessian_point(x4, objfun(x4))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3, x4], objfun, 0, mystr="After adding x4")

        # Append beyond what we are supposed to
        x5 = np.array([-1.0, -1.0])
        model.append_hessian_point(x5, objfun(x5))
        self.generic_model_check(model, npt, [x0, x1, x2], [x4, x5], objfun, 0, mystr="After adding x5")

        # Clear Hessian points
        model.clear_hessian_points()
        self.generic_model_check(model, npt, [x0, x1, x2], [], objfun, 0, mystr="After clearing")

        # Replace x0 with x4 (x0 should move to Hessian)
        model.change_simplex_point(0, x4, objfun(x4), check_not_kopt=False)
        self.generic_model_check(model, npt, [x4, x1, x2], [x0], objfun, 0, mystr="After dropping x4")

        # Remove then add simplex point
        model.remove_simplex_point(1, check_not_kopt=True)
        self.generic_model_check(model, npt, [x4, x2], [x0, x1], objfun, 0, mystr="After dropping x1")

        model.append_simplex_point(x3, objfun(x3))
        self.generic_model_check(model, npt, [x4, x2, x3], [x0, x1], objfun, 0, mystr="After re-adding x3")

        remove_point_with_check = lambda k: model.remove_simplex_point(k, check_not_kopt=True)
        self.assertRaises(AssertionError, remove_point_with_check, 0)
        model.remove_simplex_point(0, check_not_kopt=False, move_to_hessian=False)
        model.append_simplex_point(x5, objfun(x5))
        self.generic_model_check(model, npt, [x2, x3, x5], [x0, x1], objfun, 0, mystr="After re-adding x1")

        # Check simplex poisedness values/overall constant by using standard interpolation set
        x6 = x0 + np.array([1.0, 0.0])  # x0 is still best value here
        x7 = x0 + np.array([0.0, 1.0])
        model.clear_hessian_points()
        model.change_simplex_point(0, x0, objfun(x0), check_not_kopt=False)
        model.change_simplex_point(1, x6, objfun(x6), check_not_kopt=False)
        model.change_simplex_point(2, x7, objfun(x7), check_not_kopt=False)
        self.generic_model_check(model, npt, [x0, x6, x7], [x3, x5], objfun, 0, mystr="After setting x6, x7")
        ps = model.poisedness_of_each_simplex_point(delta=1.0)
        self.assertArrayEqual(ps, np.array([1.0+np.sqrt(2.0), 1.0, 1.0]), msg="Wrong poisedness of each point")
        ps = model.poisedness_of_each_simplex_point(d=np.array([-0.5, 1.0]))
        self.assertArrayEqual(ps, np.array([0.5, 0.5, 1.0]), msg="Wrong poisedness of each point 2")
        self.assertAlmostEqual(model.simplex_poisedness(delta), 1.0+np.sqrt(2.0), msg="Wrong overall poisedness")

        # self.assertTrue(False, msg="Everything passed")


class TestInterpFullDim_np1(InterpSetTestCase):
    def runTest(self):
        # Full-dimensional model for Rosenbrock (no Hessian)
        objfun = rosenbrock
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n = 2
        npt = n + 1
        x0 = np.array([-1.2, 1.0])
        f0 = objfun(x0)
        model = InterpSet(n, npt-n-1, x0, f0, abs_tol=-1e20, precondition=True)
        self.assertEqual(model.n, n, msg="Wrong n")
        self.assertEqual(model.p, n, msg="Wrong p")
        self.assertEqual(model.max_hess_npt, 0, msg="Wrong Hessian npt")
        self.assertFalse(model.have_hess, msg="Model doesn't have Hessian")
        self.assertEqual(model.num_hess_pts, 0, msg="Wrong num_hess_pts")
        self.assertFalse(model.simplex_factorisation_current, msg="Factorisation not up-to-date")
        self.assertArrayEqual(model.xpt(0), x0, msg="Wrong x0")
        self.assertArrayEqual(model.fval(0), f0, msg="Wrong f0")
        np.random.seed(0)
        delta = 1.0
        nf = 1
        maxfun = 10
        exit_info, nf = model.initialise_interp_simplex(delta, objfun, args, scaling_changes, nf, maxfun)
        # Extracted manually, given np.random.seed(0)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
        # for i in range(1, n+1):  # points[0] is x0
        #     print("Point %g" % i, model.points[i,:] - x0)
        x1 = x0 + delta * np.array([-0.874428832365212, -0.485153807702683])
        x2 = x0 + delta * np.array([-0.485153807702683, 0.874428832365212])
        self.assertAlmostEqual(np.linalg.norm(x1 - x0), delta, msg="distance x1-x0 not delta")
        self.assertAlmostEqual(np.linalg.norm(x2 - x0), delta, msg="distance x2-x0 not delta")
        self.assertAlmostEqual(np.dot(x1 - x0, x2 - x0), 0.0, msg="x1 and x2 not orthogonal")
        self.assertFalse(model.simplex_factorisation_current, msg="Wrong factorisation_current after init")
        self.generic_model_check(model, npt, [x0, x1, x2], [], objfun, 0, mystr="After init")
        self.assertIsNone(exit_info, msg="Init failed")
        self.assertEqual(nf, n+1, msg="Wrong nf after init")

        # Append beyond what we are supposed to (which is anything, in this instance)
        x5 = np.array([-1.0, -1.0])
        add_point = lambda x: model.append_hessian_point(x, objfun(x))
        self.assertRaises(AssertionError, add_point, x5)

        # self.assertTrue(False, msg="Everything passed")


class TestInterpFullDim_nsq(InterpSetTestCase):
    def runTest(self):
        # Full-dimensional model for Rosenbrock (full Hessian)
        objfun = rosenbrock
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n = 2
        npt = (n+1)*(n+2)//2
        x0 = np.array([-1.2, 1.0])
        f0 = objfun(x0)
        model = InterpSet(n, npt-n-1, x0, f0, abs_tol=-1e20, precondition=True)
        self.assertEqual(model.n, n, msg="Wrong n")
        self.assertEqual(model.p, n, msg="Wrong p")
        self.assertEqual(model.max_hess_npt, n*(n+1)//2, msg="Wrong Hessian npt")
        self.assertTrue(model.have_hess, msg="Model has Hessian")
        self.assertEqual(model.num_hess_pts, 0, msg="Wrong num_hess_pts")
        self.assertFalse(model.simplex_factorisation_current, msg="Factorisation not up-to-date")
        self.assertArrayEqual(model.xpt(0), x0, msg="Wrong x0")
        self.assertArrayEqual(model.fval(0), f0, msg="Wrong f0")
        np.random.seed(0)
        delta = 1.0
        nf = 1
        maxfun = 10
        exit_info, nf = model.initialise_interp_simplex(delta, objfun, args, scaling_changes, nf, maxfun)
        # Extracted manually, given np.random.seed(0)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
        # for i in range(1, n+1):  # points[0] is x0
        #     print("Point %g" % i, model.points[i,:] - x0)
        x1 = x0 + delta * np.array([-0.874428832365212, -0.485153807702683])
        x2 = x0 + delta * np.array([-0.485153807702683, 0.874428832365212])
        self.assertAlmostEqual(np.linalg.norm(x1 - x0), delta, msg="distance x1-x0 not delta")
        self.assertAlmostEqual(np.linalg.norm(x2 - x0), delta, msg="distance x2-x0 not delta")
        self.assertAlmostEqual(np.dot(x1 - x0, x2 - x0), 0.0, msg="x1 and x2 not orthogonal")
        self.assertFalse(model.simplex_factorisation_current, msg="Wrong factorisation_current after init")
        self.generic_model_check(model, npt, [x0, x1, x2], [], objfun, 0, mystr="After init")
        self.assertIsNone(exit_info, msg="Init failed")
        self.assertEqual(nf, n+1, msg="Wrong nf after init")

        # Append Hessian points
        x3 = np.array([5.0, 5.0])
        x4 = np.array([1.0, 1.0])  # best point
        x5 = np.array([-1.0, -1.0])
        model.append_hessian_point(x3, objfun(x3))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3], objfun, 0, mystr="After adding full Hessian 1")
        model.append_hessian_point(x4, objfun(x4))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3, x4], objfun, 0, mystr="After adding full Hessian 2")
        model.append_hessian_point(x5, objfun(x5))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3, x4, x5], objfun, 0, mystr="After adding full Hessian 3")

        model.change_simplex_point(1, x3, objfun(x3))
        self.generic_model_check(model, npt, [x0, x3, x2], [x4, x5, x1], objfun, 0, mystr="After adding x3")

        model.change_simplex_point(2, x4, objfun(x4))
        self.generic_model_check(model, npt, [x0, x3, x4], [x5, x1, x2], objfun, 2, mystr="After adding x4")

        # self.assertTrue(False, msg="Everything passed")


class TestInterpReducedDim_2np1(InterpSetTestCase):
    def runTest(self):
        # Reduced-dimensional model for Powell Singular
        objfun = powell_singular
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n = 4
        p = 2
        npt = 2*p+1
        x0 = np.array([3.0, -1.0, 0.0, 1.0])
        f0 = objfun(x0)
        model = InterpSet(p, npt-p-1, x0, f0, abs_tol=-1e20, precondition=True)
        self.assertEqual(model.n, n, msg="Wrong n")
        self.assertEqual(model.p, p, msg="Wrong p")
        self.assertEqual(model.max_hess_npt, p, msg="Wrong Hessian npt")
        self.assertTrue(model.have_hess, msg="Model has Hessian")
        self.assertEqual(model.num_hess_pts, 0, msg="Wrong num_hess_pts")
        self.assertFalse(model.simplex_factorisation_current, msg="Factorisation not up-to-date")
        self.assertArrayEqual(model.xpt(0), x0, msg="Wrong x0")
        self.assertArrayEqual(model.fval(0), f0, msg="Wrong f0")
        np.random.seed(0)
        delta = 1.0
        nf = 1
        maxfun = 10
        exit_info, nf = model.initialise_interp_simplex(delta, objfun, args, scaling_changes, nf, maxfun)
        # Extracted manually, given np.random.seed(0)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
        # for i in range(1, n+1):  # points[0] is x0
        #     print("Point %g" % i, model.points[i,:] - x0)
        s1 = np.array([-0.606484744183092, -0.336492087249845, -0.642070192810455, -0.326642308643197])
        s2 = np.array([-0.083779357809710, -0.866769259973379, 0.480508138821155, 0.103942280602377])
        x1 = x0 + delta * s1
        x2 = x0 + delta * s2
        self.assertAlmostEqual(np.linalg.norm(x1 - x0), delta, msg="distance x1-x0 not delta")
        self.assertAlmostEqual(np.linalg.norm(x2 - x0), delta, msg="distance x2-x0 not delta")
        self.assertAlmostEqual(np.dot(x1 - x0, x2 - x0), 0.0, msg="x1 and x2 not orthogonal")
        self.assertFalse(model.simplex_factorisation_current, msg="Wrong factorisation_current after init")
        self.generic_model_check(model, npt, [x0, x1, x2], [], objfun, 0, mystr="After init")
        self.assertIsNone(exit_info, msg="Init failed")
        self.assertEqual(nf, p + 1, msg="Wrong nf after init")

        # Some new points in x0+span(s1,s2)
        x3 = x0 + 0.5 * delta * (-3.8 * s1 + 2.3 * s2)
        x4 = x0 + 0.7 * delta * (1.7 * s1 - 0.3 * s2)  # best point so far (ends up replacing x1)
        # for k, xk in enumerate([x0, x1, x2, x3, x4]):
        #     print(k, objfun(xk))
        model.append_hessian_point(x3, objfun(x3))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3], objfun, 0, mystr="After x3")
        model.append_hessian_point(x4, objfun(x4))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3, x4], objfun, 0, mystr="After x4")

        model.change_simplex_point(1, x4, objfun(x4))
        self.generic_model_check(model, npt, [x0, x4, x2], [x4, x1], objfun, 1, mystr="After x4 v2")
        model.change_simplex_point(0, x3, objfun(x3))
        self.generic_model_check(model, npt, [x3, x4, x2], [x1, x0], objfun, 1, mystr="After x3 v2")

        # self.assertTrue(False, msg="Everything passed")


class TestInterpReducedDim_np1(InterpSetTestCase):
    def runTest(self):
        # Reduced-dimensional model for Powell Singular
        objfun = powell_singular
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n = 4
        p = 2
        npt = p+1
        x0 = np.array([3.0, -1.0, 0.0, 1.0])
        f0 = objfun(x0)
        model = InterpSet(p, npt-p-1, x0, f0, abs_tol=-1e20, precondition=True)
        self.assertEqual(model.n, n, msg="Wrong n")
        self.assertEqual(model.p, p, msg="Wrong p")
        self.assertEqual(model.max_hess_npt, 0, msg="Wrong Hessian npt")
        self.assertFalse(model.have_hess, msg="Model doesn't have Hessian")
        self.assertEqual(model.num_hess_pts, 0, msg="Wrong num_hess_pts")
        self.assertFalse(model.simplex_factorisation_current, msg="Factorisation not up-to-date")
        self.assertArrayEqual(model.xpt(0), x0, msg="Wrong x0")
        self.assertArrayEqual(model.fval(0), f0, msg="Wrong f0")
        np.random.seed(0)
        delta = 1.0
        nf = 1
        maxfun = 10
        exit_info, nf = model.initialise_interp_simplex(delta, objfun, args, scaling_changes, nf, maxfun)
        # Extracted manually, given np.random.seed(0)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
        # for i in range(1, n+1):  # points[0] is x0
        #     print("Point %g" % i, model.points[i,:] - x0)
        s1 = np.array([-0.606484744183092, -0.336492087249845, -0.642070192810455, -0.326642308643197])
        s2 = np.array([-0.083779357809710, -0.866769259973379, 0.480508138821155, 0.103942280602377])
        x1 = x0 + delta * s1
        x2 = x0 + delta * s2
        self.assertAlmostEqual(np.linalg.norm(x1 - x0), delta, msg="distance x1-x0 not delta")
        self.assertAlmostEqual(np.linalg.norm(x2 - x0), delta, msg="distance x2-x0 not delta")
        self.assertAlmostEqual(np.dot(x1 - x0, x2 - x0), 0.0, msg="x1 and x2 not orthogonal")
        self.assertFalse(model.simplex_factorisation_current, msg="Wrong factorisation_current after init")
        self.generic_model_check(model, npt, [x0, x1, x2], [], objfun, 0, mystr="After init")
        self.assertIsNone(exit_info, msg="Init failed")
        self.assertEqual(nf, p + 1, msg="Wrong nf after init")

        # Some new points in x0+span(s1,s2)
        x3 = x0 + 0.5 * delta * (-3.8 * s1 + 2.3 * s2)
        x4 = x0 + 0.7 * delta * (1.7 * s1 - 0.3 * s2)  # best point so far (ends up replacing x1)
        # for k, xk in enumerate([x0, x1, x2, x3, x4]):
        #     print(k, objfun(xk))
        add_points = lambda tmp: model.append_hessian_point(x3, objfun(x3))
        self.assertRaises(AssertionError, add_points, None)
        self.generic_model_check(model, npt, [x0, x1, x2], [], objfun, 0, mystr="After adding points")

        # self.assertTrue(False, msg="Everything passed")


class TestInterpReducedDim_nsq(InterpSetTestCase):
    def runTest(self):
        # Reduced-dimensional model for Powell Singular
        objfun = powell_singular
        args = ()  # optional arguments for objfun
        scaling_changes = None
        n = 4
        p = 2
        npt = (p+1)*(p+2)//2
        x0 = np.array([3.0, -1.0, 0.0, 1.0])
        f0 = objfun(x0)
        model = InterpSet(p, npt-p-1, x0, f0, abs_tol=-1e20, precondition=True)
        self.assertEqual(model.n, n, msg="Wrong n")
        self.assertEqual(model.p, p, msg="Wrong p")
        self.assertEqual(model.max_hess_npt, p*(p+1)//2, msg="Wrong Hessian npt")
        self.assertTrue(model.have_hess, msg="Model has Hessian")
        self.assertEqual(model.num_hess_pts, 0, msg="Wrong num_hess_pts")
        self.assertFalse(model.simplex_factorisation_current, msg="Factorisation not up-to-date")
        self.assertArrayEqual(model.xpt(0), x0, msg="Wrong x0")
        self.assertArrayEqual(model.fval(0), f0, msg="Wrong f0")
        np.random.seed(0)
        delta = 1.0
        nf = 1
        maxfun = 10
        exit_info, nf = model.initialise_interp_simplex(delta, objfun, args, scaling_changes, nf, maxfun)
        # Extracted manually, given np.random.seed(0)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
        # for i in range(1, n+1):  # points[0] is x0
        #     print("Point %g" % i, model.points[i,:] - x0)
        s1 = np.array([-0.606484744183092, -0.336492087249845, -0.642070192810455, -0.326642308643197])
        s2 = np.array([-0.083779357809710, -0.866769259973379, 0.480508138821155, 0.103942280602377])
        x1 = x0 + delta * s1
        x2 = x0 + delta * s2
        self.assertAlmostEqual(np.linalg.norm(x1 - x0), delta, msg="distance x1-x0 not delta")
        self.assertAlmostEqual(np.linalg.norm(x2 - x0), delta, msg="distance x2-x0 not delta")
        self.assertAlmostEqual(np.dot(x1 - x0, x2 - x0), 0.0, msg="x1 and x2 not orthogonal")
        self.assertFalse(model.simplex_factorisation_current, msg="Wrong factorisation_current after init")
        self.generic_model_check(model, npt, [x0, x1, x2], [], objfun, 0, mystr="After init")
        self.assertIsNone(exit_info, msg="Init failed")
        self.assertEqual(nf, p + 1, msg="Wrong nf after init")

        # Some new points in x0+span(s1,s2)
        x3 = x0 + 0.5 * delta * (-3.8 * s1 + 2.3 * s2)
        x4 = x0 + 0.7 * delta * (1.7 * s1 - 0.3 * s2)  # best point so far (ends up replacing x1)
        x5 = x0 + delta * (-3.8 * s1 + 2.3 * s2)
        # for k, xk in enumerate([x0, x1, x2, x3, x4, x5]):
        #     print(k, objfun(xk))
        model.append_hessian_point(x3, objfun(x3))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3], objfun, 0, mystr="After adding full Hessian 1")
        model.append_hessian_point(x4, objfun(x4))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3, x4], objfun, 0, mystr="After adding full Hessian 2")
        model.append_hessian_point(x5, objfun(x5))
        self.generic_model_check(model, npt, [x0, x1, x2], [x3, x4, x5], objfun, 0, mystr="After adding full Hessian 3")

        model.change_simplex_point(1, x3, objfun(x3))
        self.generic_model_check(model, npt, [x0, x3, x2], [x4, x5, x1], objfun, 0, mystr="After adding x3")

        model.change_simplex_point(2, x4, objfun(x4))
        self.generic_model_check(model, npt, [x0, x3, x4], [x5, x1, x2], objfun, 2, mystr="After adding x4")

        # self.assertTrue(False, msg="Everything passed")