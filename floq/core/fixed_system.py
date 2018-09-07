import logging
import numpy as np
from .. import linalg
from ..core import evolution as ev

class FixedSystem(object):
    """Class that defines and computes a specific time-evolution.

    The FixedSystem class serves essentially two purposes: On one hand, it
    contains all information describing a particular time-evolution problem to
    be solved numerically: hf, dhf and various parameters collected in an
    instance of FixedSystemParameters.

    On the other hand, it interfaces with the core.evolution module, computing u
    and du and performing caching and various house-keeping operations, in
    particular increasing nz if U is not unitary for a given choice.

    Methods
        u: computes u / returns already computed u
        du: computes du / returns already computed du

    Attributes:
        hf: the Fourier transformed Hamiltonian (ndarray, square)
        dhf: its derivative with respect to the controls (ndarray of square
             ndarrays), the first index running over the control parameters
        params: an instance of FixedSystemParameters
        decimals: number of decimals used internally for detecting degeneracies
        sparse: if yes, sparse matrix computations are performed
        max_nz: maximum nz"""
    def __init__(self, hf, dhf, nz, omega, t, decimals=10, sparse=True,\
                       max_nz=999):
        self.hf = hf
        self.dhf = dhf
        self.max_nz = max_nz
        # Inferred parameters
        dim = hf.shape[1]
        nc = hf.shape[0]
        np = dhf.shape[0]
        self.params = FixedSystemParameters(dim, nz, nc, np, omega, t, decimals,
                                            sparse=sparse)
        self._u = None
        self._udot = None
        self._du = None
        self._vals, self._vecs, self._phi, self._psi = None, None, None, None

    def __eq__(self, other):
        assert isinstance(other, FixedSystem)
        hf_same = np.array_equal(self.hf, other.hf)
        dhf_same = np.array_equal(self.dhf, other.dhf)
        params_same = (self.params.nz, self.params.omega, self.params.t) \
            == (other.params.nz, other.params.omega, other.params.t)
        return hf_same and dhf_same and params_same

    @property
    def u(self):
        if self._u is not None:
            return self._u
        else:
            self._compute_u()
            return self._u

    @property
    def du_dt(self):
        if self._udot is not None:
            return self._udot
        else:
            self._compute_udot()
            return self._udot

    @property
    def du_dcontrols(self):
        if self._du is not None:
            return self._du
        else:
            self._compute_du()
            return self._du

    def _compute_u(self):
        """Increase nz until U can be computed, then set U and the intermediary
        results."""
        nz_okay, results = self._test_nz()
        while not nz_okay:
            self.params.nz += 2
            logging.debug("Increased nz to {}".format(self.params.nz))
            if self.max_nz < self.params.nz:
                raise RuntimeError("NZ has grown too large: {} > {}"\
                                   .format(self.params.nz, self.max_nz))
            nz_okay, results = self._test_nz()
        self._u, self._vals, self._vecs, self._phi, self._psi = results

    def _test_nz(self):
        """Try to compute U with the current nz.

        If an error occurs, or U is not unitary,
            return [False, []]
        else
            return [True, [u, vecs, vals, phi, psi]]."""
        results = ev.get_u_and_eigensystem(self.hf, self.params)
        if linalg.is_unitary(results[0], tolerance=10**-self.params.decimals):
            return True, results
        else:
            return False, ()

    def _compute_udot(self):
        if self._u is None:
            self._compute_u()
        self._udot = ev.get_udot_from_eigensystem(self._phi, self._psi,
                                                  self._vals, self._vecs,
                                                  self.params)

    def _compute_du(self):
        if self._u is None:
            self._compute_u()
        self._du = ev.get_du_from_eigensystem(self.dhf, self._psi, self._vals,
                                              self._vecs, self.params)

class DummyFixedSystem(FixedSystem):
    """A dummy FixedSystem that can be initialised with arbitrary dimensions
    without specifying hf and dhf (mainly for testing)."""
    def __init__(self, **kwargs):
        self.params = FixedSystemParameters(**kwargs)

class FixedSystemParameters(object):
    """Hold parameters for a FixedSystem.

    - dim: the size of the Hilbert space.
    - nz: Number of Fourier components taken into account for K, and of
          Brillouin zones in the resulting set of eigenvalues.

    Derived from nz:
        - nz_min/max, the cutoff for the integers labelling the Fourier
                      components
        - k_kim = dim*nz: the size of K

    - nc: number of components in Hf
    - np: number of control parameters
    - omega: The frequency associated with the period T of the control pulse
    - t: Control duration
    - decimals: The number of decimals used for rounding when finding unique
                eigenvalues
    - sparse: If True, a sparse eigensolver will be used. Unless working with
              very small systems with < 15 zones, this should be True."""
    def __init__(self, dim=0, nz=1, nc=1, np=0, omega=1, t=1, decimals=10,\
                       sparse=True):
        self._dim = 0
        self._nz = 0
        self._nc = 0
        self.dim = dim
        self.nz = nz
        self.nc = nc
        self.np = np
        self.omega = omega
        self.t = t
        self.decimals = decimals
        self.sparse = sparse

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        if value % 2 == 0:
            raise ValueError("Number of Fourier components in the"
                             + " extended space (nz) cannot be even.")
        self._nz = value
        self.k_dim = self.dim * value
        self.nz_max = (value - 1) // 2
        self.nz_min = -self.nz_max

    @property
    def nc(self):
        return self._nc

    @nc.setter
    def nc(self, value):
        if value % 2 == 0:
            raise ValueError("Number of Fourier components of H"
                                + " cannot be even.")
        self._nc = value
        self.nc_max = (value - 1) // 2
        self.nc_min = -self.nc_max

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value
        self.k_dim = self.nz * value
