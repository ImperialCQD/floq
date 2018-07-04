import numpy as np
from ..core import fixed_system as fs
from .. import errors as er
import copy

class ParametricSystemBase(object):
    """Specifies a physical system that still has open parameters, such as the
    control amplitudes, the control duration, or other arbitrary parameters in
    the Hamiltonian.

    To function, this needs to be sub-classed, and a subclass should provide:
        _hf(controls): returning an array with the Fourier-transformed
                       Hamiltonian with the first index the Fourier index,
        _dhf(controls): returning the derivative of the Hamiltonian
                        with the first index signifying the control parameter.

    Methods:
        u(controls, t)
        du(controls, t),
    which implement basic caching and automatically keeps self.nz updated.

    Attributes:
        nz: (initial) number of Brillouin zones (should be overwritten)
        omega: frequency of the control signal (should be overwritten)
        sparse: if True, sparse matrix algebra is used (can be overwritten)
        max_nz: max nz allowed (can be overwritten)
        decimals: decimals used to check for unitarity (can be overwritten)."""
    def __init__(self, **kwargs):
        self._last_controls = None
        self._last_t = None
        # set defaults
        self.max_nz = 999
        self.sparse = True
        self.decimals = 10
        # these should be overwritten by a subclass
        self.nz = 3
        self.omega = 1.0

    def _hf(self, controls):
        raise NotImplementedError

    def _dhf(self, controls):
        raise NotImplementedError

    def u(self, controls, t):
        if self._is_cached(controls, t):
            return self._fixed_system.u
        else:
            self._set_cached(controls, t)
            u = self._fixed_system.u
            self.nz = self._fixed_system.params.nz
            return u

    def udot(self, controls, t):
        if self._is_cached(controls, t):
            return self._fixed_system.udot
        else:
            self._set_cached(controls, t)
            udot = self._fixed_system.udot
            self.nz = self._fixed_system.params.nz
            return udot

    def du(self, controls, t):
        if self._is_cached(controls, t):
            return self._fixed_system.du
        else:
            self._set_cached(controls, t)
            du = self._fixed_system.du
            self.nz = self._fixed_system.params.nz
            return du

    def _is_cached(self, controls, t):
        return self._last_t == t\
               and np.array_equal(self._last_controls, controls)

    def _set_cached(self, controls, t):
        self._last_controls = np.copy(controls)
        self._last_t = copy.copy(t)
        hf = self._hf(controls)
        dhf = self._dhf(controls)
        self._fixed_system = fs.FixedSystem(hf, dhf, self.nz, self.omega, t,
                                            decimals=self.decimals,
                                            sparse=self.sparse,
                                            max_nz=self.max_nz)

    def heff(self, controls, t):
        u = self.u(controls, t)
        udot = self.udot(controls, t)
        return 1j * np.dot(udot, np.conj(u.T))

class ParametricSystemWithFunctions(ParametricSystemBase):
    """A ParametricSystem that wraps callables hf and dhf."""
    def __init__(self, hf, dhf, nz, omega, parameters):
        """
        hf: callable hf(controls,parameters,omega)
        dhf: callable dhf(controls,parameters,omega)
        omega: 2 pi/T, the period of the Hamiltonian
        nz: number of Fourier modes to be considered during evolution
        parameters: a data structure that holds parameters for hf and dhf
        (dictionary is probably the best idea)
        """
        self.hf = hf
        self.dhf = dhf
        self.omega = omega
        self.nz = nz
        self.parameters = parameters

    def _hf(self, controls):
        return self.hf(controls, self.parameters, self.omega)

    def _dhf(self, controls):
        return self.dhf(controls, self.parameters, self.omega)
