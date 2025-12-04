#!/usr/bin/env python3

"""
Demonstration of RSDFO-Q on a larger-scale problem, showing
how to select the subspace dimension.
"""
import numpy as np
import rsdfoq

# Define the "variable dimension" objective function
# See problem 25 from More, Garbow & Hillstrom, Testing Unconstrained Optimization Software
# ACM Trans. Math. Softw. 7:1 (1981), pp. 17-41.
# (unique minimizer has objective value 0 at [1,...,1])
def var_dim(x):
    fx = 0.0
    for i in range(len(x)):
        fx += (x[i] - 1.0) ** 2
    f_extra = np.dot(np.arange(1, n + 1), x - 1)
    fx += f_extra ** 2 + f_extra ** 4
    return fx

# Desired dimension of objective function and starting point for the solver
n = 20  # any positive integer is allowed
x0 = 1.0 - np.arange(1, n+1) / n

print("Initial objective value = %g" % var_dim(x0))

# RSDFO-Q is a randomized algorithm (i.e. the process to generate
# iterates is random, even if the objective is deterministic)
# so it may sometimes be a good idea to fix the random seed
# to ensure reproducibility
np.random.seed(0)

# Desired subspace dimension, must be a value in [1, ..., len(x0)]
# The solver runs more quickly for smaller subspace dimensions, but
# usually requires fewer total objective evaluations for larger
# subspace dimensions --- select based on user preference
subspace_dim = 2

# Perform the optimization, up to a maximum computational budget
# (measured in maximum number of objective evaluations)
soln = rsdfoq.solve(var_dim, x0, fixed_block_size=subspace_dim, maxfun=2000)

# Inspect the solution
# Note: the actual minimizer is stored in soln.x
print(soln)
