======================================================
RSDFO-Q: Randomized Subspace DFO with Quadratic Models
======================================================

.. image::  https://github.com/lindonroberts/rsdfoq/actions/workflows/unit_tests.yml/badge.svg
   :target: https://github.com/lindonroberts/rsdfoq/actions
   :alt: Build Status

.. image::  https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: GNU GPL v3 License

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
You can install RSDFO-Q by cloning this repository and installing with pip:

.. code-block:: bash

    $ git clone https://github.com/lindonroberts/rsdfoq.git
    $ cd rsdfoq
    $ ls                     <-- check for pyproject.toml
    $ pip install -e .

RSDFO-Q requires NumPy, SciPy and pandas, but these will be installed automatically
if they are not already available.

Usage
-----
Examples for how to use RSDFO-Q may be found in the :code:`examples` directory.
