#!/usr/bin/env python3

"""
Basic test of RSDFO-Q - minimize the 2D Rosenbrock function

Note: RSDFO-Q is mostly designed for large-scale problems
(i.e. many decision variables, typically O(100) or greater)
but this script is just to demonstrate the basic usage.
"""
import numpy as np
import rsdfoq

# Define the objective function
# (unique minimizer has objective value 0 at [1,1])
def rosenbrock(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

# Define the starting point for the solver
x0 = np.array([-1.2, 1.0])

# RSDFO-Q is a randomized algorithm (i.e. the process to generate
# iterates is random, even if the objective is deterministic)
# so it may sometimes be a good idea to fix the random seed
# to ensure reproducibility
np.random.seed(0)

# Perform the optimization, up to a maximum computational budget
# (measured in maximum number of objective evaluations)
soln = rsdfoq.solve(rosenbrock, x0, maxfun=1000)

# Inspect the solution
# Note: the actual minimizer is stored in soln.x
print(soln)
