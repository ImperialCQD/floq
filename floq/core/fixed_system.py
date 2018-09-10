import logging
import numpy as np
from .. import linalg
from . import evolution

class FixedSystem:
    """
    Class that defines and computes a specific time-evolution.  It currently
    provides a layer between `floq.System` and `floq.core.evolution`, although
    this may be liable to change in the future.
    """
    def __init__(self, hamiltonian, dhamiltonian, n_zones, omega, time,
                       decimals=10, sparse=True, max_zones=999):
        """
        Arguments --
        hamiltonian: 3D np.array of complex as in a list of 2D arrays --
            The shape is
                (n_fourier, dimension, dimension)
            where `n_fourier` is the total number of Fourier modes, and
            `dimension` is the dimension of the original problem Hamiltonian.
            The first shape argument must be odd, because if the maximum
            absolute value of a Fourier mode considered is `m`, then the index
            runs as
                -m, -m + 1, ..., 0, 1, ..., m.

        dhamiltonian: 4D np.array of complex --
            This is an array containing the derivative of `hamiltonian` with
            respect to each of the control parameters in turn.  The shape is
                (n_controls, n_fourier, dimension, dimension),
            so the first index runs over the controls, and the other three are
            the same as `hamiltonian`.

        nz: odd int --
            The initial number of Brillouin zones to consider.  This will be
            increased internally up to `max_nz` if necessary to maintain
            unitarity.

        omega: float --
            The angular frequency of the Fourier transformation (so it's 2pi
            divided by the period).

        t: float --
            The time to evaluate the time evolution operator at.

        decimals: int --
            The number of decimal places of precision to use when comparing the
            created operator's unitary properties with the identity.

        sparse: bool --
            Whether to use sparse matrix algebra when diagonalising `K`.

        max_nz: int --
            The maximum number of Brillouin zones to use.  If the created
            operator is still not unitary once this value has been reached, a
            `RuntimeError` will be raised.
        """
        self.hamiltonian = hamiltonian
        self.dhamiltonian = dhamiltonian
        self.max_zones = max_zones
        # Inferred parameters
        dimension = hamiltonian.shape[1]
        n_fourier = hamiltonian.shape[0]
        n_controls = dhamiltonian.shape[0]
        self.parameters = FixedSystemParameters(dimension, n_zones, n_fourier,
                                                n_controls, omega, time,
                                                decimals, sparse=sparse)
        for x in ('u', 'du_dt', 'du_dcontrols', 'vals', 'vecs', 'phi', 'psi'):
            setattr(self, f"_FixedSystem__{x}", None)

    def __compute_eigensystem_and_u(self):
        while self.__u is None:
            if self.parameters.nz > self.max_zones:
                raise RuntimeError(
                    "The number of Brillouin zones became too large before the"
                    + " time-evolution operator became unitary.  Currently at"
                    + f" {self.parameters.nz}, but the maximum was"
                    + f" {self.max_zones}.")
            results = evolution.get_u_and_eigensystem(self.hamiltonian,
                                                      self.parameters)
            if linalg.is_unitary(results[0], self.parameters.decimals):
                self.__u, self.__vals, self.__vecs, self.__phi, self.__psi =\
                    results
            else:
                self.parameters.nz += 2
                logging.debug(f"Increased nz to {self.parameters.nz}")

    @property
    def u(self):
        if self.__u is not None:
            return self.__u
        self.__compute_eigensystem_and_u()
        return self.__u

    @property
    def du_dt(self):
        if self.__du_dt is not None:
            return self.__du_dt
        self.__compute_eigensystem_and_u()
        self.__du_dt = evolution.get_du_dt_from_eigensystem(
                    self.__phi, self.__psi, self.__vals, self.__vecs,
                    self.parameters)
        return self.__du_dt

    @property
    def du_dcontrols(self):
        if self.__du_dcontrols is not None:
            return self.__du_dcontrols
        self.__compute_eigensystem_and_u()
        self.__du_dcontrols = evolution.get_du_dcontrols_from_eigensystem(
                self.dhamiltonian, self.__psi, self.__vals, self.__vecs,
                self.parameters)
        return self.__du_dcontrols

class FixedSystemParameters:
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
