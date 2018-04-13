# FLOQ

Originally written by Marcel Langer, and this version is forked from [his
Github](http://www.github.com/sirmarcel/floq).

## A Python module for smooth robust quantum control

This repository is the home of `floq`, a Python module that implements the
smooth robust quantum control approach as laid out in [*Smooth optimal control
with Floquet theory* by Bartels and Mintert](https://arxiv.org/abs/1205.5142v2).
This implementation was written as part of my Master thesis, which can be found
[here](http://marcel-langer.com/ma). It explains the background and terminology
of the code in some detail. If the following documentation is unclear, the
thesis might be a useful backup, or you can drop me a line at me@(username).com

All of this is a work in progress! If you are not involved in this project
personally, you might want to hold out for now, until the API has settled and
proper documentation is in place.

In the following, some familiarity with quantum control and Python will be
assumed — sorry. Please don’t hesitate to ask questions.

## Installation

Currently, `floq` is not distributed through `pip` or `conda` and has to be
installed manually by following these steps, assuming Python 3.

Make sure the required packages are installed on your computer, which you
probably want to do with [`conda`](https://conda.io/docs/get-started.html)

```conda install numba numpy scipy nose mock```

Then, clone this repository to your machine:

```git clone git@github.com:ImperialCQD/floq```

In order to get Python to recognise this manually downloaded package as a
module, you should add the folder the repository is in to the Python path
environment variable by putting the following line in your
```.profile/.bashrc/.zshrc```:

```export PYTHONPATH="$PYTHONPATH:/path/to/floq"```

Now verify that ```floq``` is working!

```cd /path/to/floq nosetests```

Should yield something like

```
.............................................................................................
---------------------------------------------------------------------- Ran 93
tests in 9.052s

OK
```

If you see this message, ```floq``` has been installed properly, and should work
correctly! You can now use it like any other Python module.

(Currently, the test ```test_nz_does_not_blow_up_one``` does fail on occasion —
I am not sure why, it’s on the todo list!)

## Overview

`floq` provides three distinct capabilities:

- An implementation of a Floquet theory solver for the Schrödinger equation,
  which works in the frequency domain for periodic Hamiltonians,
- A framework to implement the components necessary to solve quantum control
  problems, and finally
- Reference implementations using this framework, for instance common
  fidelities, and in particular extensions to handle robust control problems.

The overall goal of this is to allow you to focus on quickly implementing the
components specific to your particular control problem, and using the
infrastructure provided by `floq` to tie them together.

In this context, a quantum control problem is described by the following parts:

- The quantum system to be controlled itself, implemented as a subclass of
  `ParametricSystemBase`, which fundamentally only needs to supply a method to
  compute the Hamiltonian of the system,
- A fidelity (subclass of `FidelityBase`) measuring how well the unitary for a
  given system and set of controls implements the intended dynamics, and
- An optimisation method (subclass of `OptimizerBase`), defining how an
  optimised set of control parameters is obtained.

For robust control problems, the fidelity is replaced with an ensemble fidelity,
and an entire ensemble of random `ParametricSystem`s is used rather than a
specific one.

## Usage

Let’s now look at some elementary examples on how to use `floq` in practice. 

### Control of a spin system

For this example, we will use the system that originally prompted the
development of smooth optimal quantum control in [*Smooth optimal quantum
control for robust solid state spin magnetometry* by Nöbauer et
al.](https://arxiv.org/abs/1412.5051): An NV centre, simplified to a two-level
system. 

(This can be easily expanded to an ensemble of such systems in order to solve
the robust control problem of controlling the ensemble with inhomogeneous
broadening and amplitude attenuation.)

The Hamiltonian of the system is

```
H(t) = w/2 s_z + 1/2 sum_k^ncomp (a_k s_x + b_k s_y) sin(k omega t)
```

Where ```s_i``` is short for the i-th Pauli matrix. The control pulse has an
angular frequency ```omega``` and the control parameters are the ```2*ncomp```
numbers ```a_k``` and ```b_k```. In the code, they are treated as a vector
```[a1, b1, a2, b2, ... a_ncomp, b_ncomp]```. 

This particular system is included in `floq`, its implementation is found in
`floq/systems/spins.py` and can serve as an instructional example on how to
implement custom quantum systems — we will come back to this task in a bit. 

#### Basics

For now, in a new file, let’s import numpy and the system:

```python
import numpy as np
from floq.systems.spins import SpinSystem
```

Now, we create an instance of SpinSystem with some particular parameters, for
instance 3 components of the control signal, no amplitude attenuation (factor
1.0), a small detuning and a frequency of 1 MHz for the control signal:

```python
spin = SpinSystem(3, 1.0, 0.01, 2*np.pi)
```

Now, computing the unitary for some random control parameters  and a time t=0.5
microsecond (a half-period of the control pulse!) is easy:

```python
ctrl = np.random.rand(3*2)
print(spin.u(ctrl, 0.5))
```

Similarly, the gradient of the unitary can be computed with a simple command:

```python
print(spin.du(ctrl, 0.5))
```

#### Control

Having computed the unitary and its gradient for some particular controls, let’s
now try to find a way to induce a particular kind of dynamics. 

For instance, let’s attempt to find a way to transition from one computational
basis state to the other:

```python
inital = np.array([1+0j, 0+0j])
final = np.array([0j, 1+0j])
```

For this, we need to import some additional components:

```python
from floq.optimization.fidelity import TransferDistance
from floq.optimization.optimizer import SciPyOptimizer
```

The transfer distance computes the inner product between the final state and U
|initial>, precisely speaking 1-|<fin| u |init>|^2. Minimising it therefore
means finding a set of controls that induce a unitary that accomplishes this
transition.

Initialise this fidelity with:

```python
fid = TransferDistance(spin, 0.5, final, inital)
```

Computing the fidelity for the random `ctrl` initialised earlier is easy, and
should yield a number > 0:

```python
print(fid.f(ctrl))
```

Finally, we create an optimiser (by default, ```SciPyOptimizer``` will use the
```scipy.minimize``` methods, in particular the BFGS method):

```python
opt = SciPyOptimizer(fid, ctrl)
```

It is initialised with the fidelity to be optimised, and a starting set of
controls, which are just random in this case.

And run it:
 
```python
print(opt.optimize())
```

The optimisation should run quickly, and produce (the ‘x’ line in the output) a
set of controls inducing the desired dynamics.

For reference, this is the full script required to run the quantum control
simulation:

```python
import numpy as np
from floq.systems.spins import SpinSystem
from floq.optimization.fidelity import TransferDistance
from floq.optimization.optimizer import SciPyOptimizer

spin = SpinSystem(3, 1.0, 0.01, 2*np.pi)
ctrl = np.random.rand(3*2)
inital = np.array([1+0j, 0+0j]) final = np.array([0j, 1+0j])
fid = TransferDistance(spin, 0.5, final, inital)
opt = SciPyOptimizer(fid, ctrl)
print(opt.optimize())
```

#### Implementing a custom quantum system

Quantum systems are implemented as sub-classes of the ```ParametricSystemBase```
class, which is defined in ```floq/systems/parametric_system.py```. Since
```floq``` is built to take care of computing the unitary and its gradient, a
custom quantum system only needs to provide the Hamiltonian and its gradient. 

However, due to the way Floquet theory operates, and due to the particular
implementation in the present case, the Hamiltonian is expected to be supplied
in *Fourier-transformed form*, and it must additionally possess a finite number
of components in Fourier space. These components are indexed with the following
notation, defining

```
H(nu) = (1/T) int_0^T dt H(t) exp(-i nu omega t)
```

Where ```nu``` is an integer. 

In practice, an implementation should look like this: 

```python

from floq.systems.parametric_system import ParametricSystemBase

class MySystem(ParametricSystemBase):
    def __init__(self, a, b, omega):
        super(MySystem, self).__init__()
        self.a = a self.b = b self.omega = omega

    def _hf(self, controls):
        # your code to return hf

    def _dhf(self, controls):
        # your code to return dhf
```

The ```hf``` and ```dhf``` should be numpy arrays of the following form:

```hf```: An ```nc x dim x dim``` array, where ```nc``` is the total number of
Fourier components treated, and ```dim``` is the dimension of the Hilbert space.

The first index should run through the Fourier components as defined above, from
negative to positive indices. ```floq``` implicitly expects this to be
*symmetric*, so the indices should run from -(nc-1)/2 through 0 to +(nc-1)/2.
Also note that this Fourier transform is with respect to the frequency of the
*control signal*, so the Hamiltonian needs to be brought into a form appropriate
for this — usually by some type of Rotating-Wave-Approximation.

```dhf```: An ```np x nc x dim x dim``` where ```np``` is the number of control
parameters (```2*ncomp``` in the spin example from above).

This should be the gradient of the Hamiltonian, where the first index now runs
through the control parameters, and each such indexed entry has the form of
```hf```.

Note that if a gradient-free optimisation method is employed, you can get away
with not implementing ```dhf```!

A fully implemented example can be found in ```floq.systems.spins```, where
```SpinSystem``` implements the NV centre system used previously. Soon, an
example project using ```floq``` will be made available, I’ll then update this
readme!
