"""Provide templates and implementations for FidelityComputer class, which wraps
a ParametricSystem and computes f and df for given controls."""
import logging
from ..linalg import d_operator_distance, operator_distance
from ..linalg import transfer_distance, d_transfer_distance
import numpy as np

class FidelityBase(object):
    """Defines how to calculate a fidelity and its gradient.

    This is a base class and needs to be sub-classed. The base class implements
    common functionality, such as counting iterations or handling optional
    penalty terms not arising directly from the fidelity.

    Sub-classes should implement:
        _f(controls_and_t)
        _df(controls_and_t).

    Sub-classes can optionally implement:
        penalty(controls_and_t)
        d_penalty(controls_and_t),
        _iterate(controls_and_t), which gets called on each iteration.

    The __init__ should take the form __init__(self, system, **kwargs)
    for compatibility with EnsembleFidelity.

    Methods:
        f(controls_and_t): returns a real number, the fidelity,
        df(controls_and_t): returns its gradient,
        iterate(controls_and_t): expected to be called after each iteration by
                                 an Optimizer.

    Attributes:
        system: the system (or ensemble) under consideration.
        iterations: count of iterations."""
    def __init__(self, system):
        self.system = system
        self.iterations = 0

    def f(self, *args, **kwargs):
        return self._f(*args, **kwargs) + self.penalty(*args, **kwargs)

    def df(self, *args, **kwargs):
        return self._df(*args, **kwargs) + self.d_penalty(*args, **kwargs)

    def iterate(self, *args, **kwargs):
        """Gets called by the Optimizer after each iteration. Increases the
        iteration count self.iterations, and calls the (optional) _iterate
        method."""
        self.iterations += 1
        self._iterate(*args, **kwargs)
        f = self.f(*args, **kwargs)
        logging.info("Currently at iteration {} and f={}"\
                     .format(self.iterations, f))

    def reset_iterations(self):
        self.iterations = 0

    def _f(self, *args, **kwargs):
        raise NotImplementedError

    def _df(self, *args, **kwargs):
        raise NotImplementedError

    def _iterate(self, *args, **kwargs):
        pass

    def penalty(self, *args, **kwargs):
        return 0.0

    def d_penalty(self, *args, **kwargs):
        return 0.0

class EnsembleFidelity(FidelityBase):
    """With a given Ensemble, and a FidelityComputer, calculate the average
    fidelity over the whole ensemble."""
    def __init__(self, ensemble, fidelity, **kwargs):
        super().__init__(ensemble)
        self.fidelities = [fidelity(sys, **kwargs) for sys in ensemble.systems]

    def _f(self, *args, **kwargs):
        return np.mean([fid.f(*args, **kwargs) for fid in self.fidelities])

    def _df(self, *args, **kwargs):
        return np.mean([fid.df(*args, **kwargs) for fid in self.fidelities],
                       axis=0)

class OperatorDistance(FidelityBase):
    """Calculate the operator distance (see core.fidelities for details) for a
    given ParametricSystem and a fixed pulse duration t."""
    def __init__(self, system, t, target):
        super().__init__(system)
        self.t = t
        self.target = target

    def _f(self, *args, **kwargs):
        return operator_distance(self.system.u(self.t, *args, **kwargs),
                                 self.target)

    def _df(self, *args, **kwargs):
        u = self.system.u(self.t, *args, **kwargs)
        du = self.system.du_dcontrols(self.t, *args, **kwargs)
        return d_operator_distance(u, du, self.target)

class TransferDistance(FidelityBase):
    """Calculate the state transfer fidelity between two states |initial> and
    |final> (see core.fidelities for details) for a given ParametricSystem and a
    fixed pulse duration t."""
    def __init__(self, system, t, initial, final):
        super().__init__(system)
        self.t = t
        self.initial = initial
        self.final = final

    def _f(self, *args, **kwargs):
        u = self.system.u(self.t, *args, **kwargs)
        return transfer_distance(u, self.initial, self.final)

    def _df(self, *args, **kwargs):
        u = self.system.u(self.t, *args, **kwargs)
        du = self.system.du(self.t, *args, **kwargs)
        return d_transfer_distance(u, du, self.initial, self.final)
