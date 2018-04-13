"""Provide templates and implementations for FidelityComputer class, which wraps
a ParametricSystem and computes f and df for given controls."""
import logging
from ..core.fidelities import d_operator_distance, operator_distance
from ..core.fidelities import transfer_distance, d_transfer_distance
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

    def f(self, controls_and_t):
        return self._f(controls_and_t) + self.penalty(controls_and_t)

    def df(self, controls_and_t):
        return self._df(controls_and_t) + self.d_penalty(controls_and_t)

    def iterate(self, controls_and_t):
        """Gets called by the Optimizer after each iteration. Increases the
        iteration count self.iterations, and calls the (optional) _iterate
        method."""
        self.iterations += 1
        self._iterate(controls_and_t)
        f = self.f(controls_and_t)
        logging.info("Currently at iteration {} and f={}"\
                     .format(self.iterations, f))

    def reset_iterations(self):
        self.iterations = 0

    def _f(self, controls_and_t):
        raise NotImplementedError

    def _df(self, controls_and_t):
        raise NotImplementedError

    def _iterate(self, controls_and_t):
        pass

    def penalty(self, controls_and_t):
        return 0.0

    def d_penalty(self, controls_and_t):
        return 0.0

class EnsembleFidelity(FidelityBase):
    """With a given Ensemble, and a FidelityComputer, calculate the average
    fidelity over the whole ensemble."""
    def __init__(self, ensemble, fidelity, **params):
        super(EnsembleFidelity, self).__init__(ensemble)
        self.fidelities = [fidelity(sys, **params) for sys in ensemble.systems]

    def _f(self, controls_and_t):
        return np.mean([fid.f(controls_and_t) for fid in self.fidelities])

    def _df(self, controls_and_t):
        return np.mean([fid.df(controls_and_t) for fid in self.fidelities],
                       axis=0)

class OperatorDistance(FidelityBase):
    """Calculate the operator distance (see core.fidelities for details) for a
    given ParametricSystem and a fixed pulse duration t."""
    def __init__(self, system, t, target):
        super(OperatorDistance, self).__init__(system)
        self.t = t
        self.target = target

    def _f(self, controls):
        return operator_distance(self.system.u(controls, self.t), self.target)

    def _df(self, controls):
        u = self.system.u(controls, self.t)
        du = self.system.du(controls, self.t)
        return d_operator_distance(u, du, self.target)

class TransferDistance(FidelityBase):
    """Calculate the state transfer fidelity between two states |initial> and
    |final> (see core.fidelities for details) for a given ParametricSystem and a
    fixed pulse duration t."""
    def __init__(self, system, t, initial, final):
        super(TransferDistance, self).__init__(system)
        self.t = t
        self.initial = initial
        self.final = final

    def _f(self, controls):
        u = self.system.u(controls, self.t)
        return transfer_distance(u, self.initial, self.final)

    def _df(self, controls):
        u = self.system.u(controls, self.t)
        du = self.system.du(controls, self.t)
        return d_transfer_distance(u, du, self.initial, self.final)
