======================================================
RSDFO-Q: Randomized Subpsace DFO with Quadratic Models
======================================================

A Python package for general minimization, where derivatives
are not available, using random subspaces.
For a description of this algorithm, see `this paper <https://arxiv.org/abs/2412.14431>`_.

For lower-dimensional problems, consider using the more actively
maintained `Py-BOBYQA <https://github.com/numericalalgorithmsgroup/pybobyqa>`_.

Citation
--------
If you use RSDFO-Q in an academic work, please cite the following paper:

C. Cartis and L. Roberts, Randomized Subspace Derivative-Free Optimization with
Quadratic Models and Second-Order Convergence. *Optimization Methods and Software*,
to appear.

A preprint version of this paper can be `found on arXiv <https://arxiv.org/abs/2412.14431>`_.

Installation
------------
The easiest way to install RSDFO-Q is using PyPI:

.. code-block:: bash

    $ pip install rsdfoq

Alternatively, you can clone this repository and, assuming your working
directory is the top-level directory of this repository (i.e. running :code:`ls`
shows :code:`pyproject.toml`), and install RSDFO-Q locally using

.. code-block:: bash

    $ pip install -e .

RSDFO-Q requires NumPy, SciPy and pandas.

Usage
-----
Examples for how to use RSDFO-Q may be found in the :code:`examples` directory.
