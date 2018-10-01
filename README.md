# floq

A Python module for smooth robust quantum control of periodic Hamiltonians.

The original author was Marcel Langer, who started this module as part of
[his Master's thesis](http://marcel-langer.com/ma).  It is now maintained by
[Jake Lishman](https://www.github.com/jakelishman) as part of the Imperial
College London Controlled Quantum Dynamics group, and you can contact me at
my Imperial email address jake.lishman16@imperial.ac.uk.

The theory behind this module can be found in Marcel's thesis, and in the paper
_Smooth optimal control with Floquet theory_ by Björn Bartels and Florian
Mintert ([arXiv](https://arxiv.org/abs/1205.5412),
[journal](https://doi.org/10.1103/PhysRevA.88.052315)).


## Installation

`floq` is not available through `pip` or `conda`, and must be installed
manually, by cloning the repository and adding the resulting directory to the
`PYTHONPATH` environment variable (or otherwise making the inner `floq` folder
visible to the Python search path).

The requirements are listed in the file `requirements.txt` in the root of the
repository.  `nose` and `mock` are only required to run the tests, and are not
needed for a regular installation.  `floq` suppports only Python 3.

If you have installed the test requirements, you can run them by navigating to
the folder that you cloned the repository into, and running `nosetests`.  All
tests should pass.


## Overview

The main use case of `floq` is to calculate the time-evolution operator for a
periodic Hamiltonian, and the evolution operators derivatives with respect to
both time and any parameters the Hamiltonian is a function of.  The derivative
of the operator with respect to the controls allows us to use gradient-based
methods for optimal control.

There are a couple of examples in the `examples/` folder, and there is more help
available in the docstrings of the code.  Try calling `help()` on classes and
functions to find out more.

The base class is `floq.System`.  This can be instantiated with just a
Hamiltonian (see the `help()` for details, and see Marcel's thesis for details
of the Fourier transformation), but the derivatives of the Hamiltonian can also
optionally be passed.
