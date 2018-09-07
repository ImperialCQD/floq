import numpy as np
import abc
import copy
from . import core

class System:
    """
    The base `floq` system class, providing methods to calculate the
    time-evolution operator and its derivatives for Fourier-transformed
    Hamiltonians.

    The base methods are `u()`, `du_dcontrols()` and `du_dt()`.  The convenience
    function `h_effective()` is also provided.
    """
    def __init__(self, hamiltonian, dhamiltonian, nz=3, omega=1.0,
                       max_nz=999, sparse=True, decimals=10, cache=True):
        """
        Arguments --
        hamiltonian: (controls: array_like) -> 3D np.array of complex --
            A function which takes an array of its control parameters and
            returns a Fourier representation of the matrix.  The shape of the
            output array is
                (modes, N, N),
            where `modes` is the number of Fourier modes (relative to the
            principle frequency `omega`) and `N` is the dimension of the
            Hamiltonian.  The number `modes` must always be an odd number,
            because if the absolute value of the maximum frequency is
            `m * omega` for integer `m`, then the modes are
                -m, -m + 1, -m + 2, ..., 0, 1, ..., m.
            This is also the order that the first dimension of the output array
            must run over - `hamiltonian(controls)[0]` should be an `N * N`
            matrix containing the Fourier mode corresponding to `-m * omega`.

        dhamiltonian: (controls: array_like) -> 4D np.array of complex --
            A function which takes an array of its control parameters and
            returns the derivatives of the Fourier representation of the matrix
            with respect to each of the control parameters in turn.  The shape
            is
                (ncontrols, modes, N, N),
            so the first index runs over the controls, and the remaining indices
            are the same as the output of `hamiltonian()`.

        nz: odd int > 0 --
            The initial number of 'Brillouin zones' to be considered.

        omega: float > 0 --
            The numerical value of the principle frequency (the frequency that
            the Fourier transform is with respect to).

        max_nz: odd int > 0 --
            The maximum size that `nz` should be allowed to grow to.  If the
            calculations try to raise `nz` above this value, a
            `floq.NZTooLargeError` will be raised.

        sparse: bool -- Whether to use sparse matrix algebra.

        decimals: int --
            The number of decimal places to be considered when checking whether
            the evolution operators have remained unitary.  A larger value is a
            tighter constraint.

        cache: bool --
            Whether to cache the results of calculations.  This defaults to
            `True`, and it's typically only useful to set this to `False` for
            timing purposes.  Only the last used controls, time and additional
            arguments are cached, so there is no real memory impact even when
            `True`.
        """
        self.__controls = None
        self.__t = None
        self.__args = None
        self.__kwargs = None
        self.__fixed = None
        self.cache = cache
        self.sparse = sparse
        self.decimals = decimals
        self.nz = nz
        self.max_nz = max_nz
        self.omega = omega
        self.hamiltonian = hamiltonian
        self.dhamiltonian = dhamiltonian

    def __update_if_required(self, controls: np.array, t: float, args, kwargs):
        """
        Update the underlying `FixedSystem` if the current version was created
        with a different set of control or time parameters.
        """
        if self.cache\
           and self.__t == t\
           and np.array_equal(self.__controls, controls)\
           and np.array_equal(self.__args, args)\
           and self.__kwargs == kwargs:
            return
        hamiltonian = self.hamiltonian(controls, *args, **kwargs)
        dhamiltonian = self.dhamiltonian(controls, *args, **kwargs)
        self.__fixed = core.FixedSystem(hamiltonian, dhamiltonian, self.nz,
                                        self.omega, t,
                                        decimals=self.decimals,
                                        sparse=self.sparse,
                                        max_nz=self.max_nz)
        self.__controls = np.copy(controls)
        self.__t = t
        self.__args = copy.copy(args)
        self.__kwargs = kwargs.copy()

    def u(self, controls: np.array, t: float, *args, **kwargs):
        """
        Calculate the time evolution operator of the stored Hamiltonian.
        """
        self.__update_if_required(controls, t, args, kwargs)
        return self.__fixed.u

    def du_dt(self, controls: np.array, t: float, *args, **kwargs):
        """
        Calculate the derivative of the time-evolution operator with respect to
        time.
        """
        self.__update_if_required(controls, t, args, kwargs)
        return self.__fixed.du_dt

    def du_dcontrols(self, controls: np.array, t: float, *args, **kwargs):
        """
        Calculate the derivatives of the time-evolution operator with respect to
        each of the control parameters in turn.
        """
        self.__update_if_required(controls, t, args, kwargs)
        return self.__fixed.du_dcontrols

    def h_effective(self, controls: np.array, t: float, *args, **kwargs):
        u = self.u(controls, t, *args, **kwargs)
        du_dt = self.du_dt(controls, t, *args, **kwargs)
        return 1j * (du_dt @ np.conj(u.T))

class EnsembleBase(abc.ABC):
    """
    Specifies an ensemble of `floq.System`s.  This class is intended to
    serve as a container for a list of `System`s, and a convenient way
    of initialising them.  This is base class defining the API and cannot be
    used, it needs to be sub-classed, and a sub-class needs to provide a
    property called 'systems'.
    """
    @property
    @abc.abstractmethod
    def systems(self):
        raise NotImplementedError

def rabi(up_energy, down_energy, **kwargs):
    def hamiltonian(controls):
        out = np.zeros((3, 2, 2), dtype=np.complex128)
        out[0, 1, 0] = out[2, 0, 1] = controls[0]
        out[1, 0, 0] = up_energy
        out[1, 1, 1] = down_energy
        return out
    deriv = np.zeros((1, 3, 2, 2), dtype=np.complex128)
    deriv[0, 0, 1, 0] = out[0, 2, 0, 1] = 1.0
    def dhamiltonian(controls):
        return deriv
    if 'nz' not in kwargs:
        kwargs['nz'] = 11
    return System(hamiltonian, dhamiltonian, **kwargs)

def _spin_hamiltonian(ncomp, freq, controls):
    """Assemble hf for one spin, given a detuning freq, the control amplitudes
    controls, and ncomp components of the control pulse."""
    nc = 2 * ncomp + 1  # number of components in hf
    hf = np.zeros((nc, 2, 2), dtype=np.complex128)
    for k in range(ncomp):
        # Controls are ordered in reverse compared to how they are placed in hf
        a = controls[-2 * k - 2]
        b = controls[-2 * k - 1]
        # The controls are placed symmetrically around
        # the centre of hf, so we can place them at the
        # same time to save us some work!
        hf[k,:,:] = np.array([[0.0, 0.25 * (1j * a + b)],
                              [0.25 * (1j * a - b), 0.0]])
        hf[-k - 1,:,:] = np.array([[0.0, -0.25 * (1j * a + b)],
                                   [0.25 * (-1j * a + b), 0.0]])
    # Set centre (with Fourier index 0)
    hf[ncomp] = np.array([[0.5 * freq, 0.0],
                          [0.0, -0.5 * freq]], dtype=np.complex128)
    return hf

def _spin_d_hamiltonian(ncomp):
    """Assemble dhf for one spin given ncomp components in the control pulse."""
    nc = 2 * ncomp + 1
    np_ = 2 * ncomp
    dhf = np.zeros([np_, nc, 2, 2], dtype=np.complex128)
    for k in range(ncomp):
        i_a = -2 * k - 2
        i_b = -2 * k - 1
        dhf[i_a, k, :, :] = np.array([[0.0, 0.25j],
                                      [0.25j, 0.0]])
        dhf[i_a, -k-1, :, :] = np.array([[0.0, -0.25j],
                                         [-0.25j, 0.0]])
        dhf[i_b, k, :, :] = np.array([[0.0, 0.25],
                                      [-0.25, 0.0]])
        dhf[i_b, -k-1, :, :] = np.array([[0.0, -0.25],
                                         [0.25, 0.0]])
    return dhf

def spin(n_components, amplitude, frequency, **kwargs):
    def hamiltonian(controls):
        return _spin_hamiltonian(n_components, frequency, amplitude * controls)
    deriv = _spin_d_hamiltonian(n_components)
    def d_hamiltonian(controls):
        return deriv
    return System(hamiltonian, d_hamiltonian, **kwargs)

class SpinEnsemble(EnsembleBase):
    """
    A system of n non-interacting spins, where each spin is described by the
    Hamiltonian
        H(t) = w/2 s_z + 1/2 sum_k (a_k s_x + b_k s_y) sin(k omega t).

    Commonly, the values for w will be set slightly different for each spin, and
    a_k and b_k will be multiplied by some attenuation factor for each spin.
    Fidelities etc. will then be computed as ensemble averages."""
    def __init__(self, n_systems, n_components, omega, frequencies, amplitudes):
        """Initialise a SpinEnsemble instance with
          - n: number of spins
          - ncomp: number of components in the control pulse
           -> hf will have nc = 2*comp+1 components
          - omega: base frequency of control pulse
          - freqs: vector of n frequencies
          - amps: vector of n amplitudes."""
        self.n_systems = n_systems
        self.n_components = n_components
        self.omega = omega
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.nz = np.full((self.n_systems,), 3, dtype=np.int32)
        self.__systems = [spin(n_components, a, f, omega=omega)\
                          for a, f in zip(amplitudes, frequencies)]

    @property
    def systems(self):
        return self.__systems

class RandomisedSpinEnsemble(SpinEnsemble):
    """A system of n non-interacting spins, where each spin is described by the
    Hamiltonian
        H(t) = w/2 s_z + 1/2 sum_k (a_k s_x + b_k s_y) sin(k omega t).

    The n spins will be instantiated with randomised detunings and
    amplitudes, distributed as follows:
    - frequencies: normal distribution with given FWHM (2 sqrt(2 ln 2) sigma)
                   and mean 0,
    - amplitudes: Uniform distribution with given width around 1.0."""
    def __init__(self, n, ncomp, omega, fwhm, amp_width):
        """Initialise a SpinEnsemble instance with
            - n: number of spins
            - ncomp: number of components in the control pulse
             -> hf will have nc = 2*comp+1 components
            - omega: base frequency of control pulse
            - fwhm: full width at half-max of the Gaussian distribution of
                    detunings
            - amp_width: amplitudes will be drawn from a uniform distribution
                         around 1 with this width."""
        sigma = fwhm / 2.35482
        freqs = np.random.normal(loc=0.0, scale=sigma, size=n)
        amps = amp_width * (2 * np.random.rand(n) - 1) + np.ones(n)
        super().__init__(n, ncomp, omega, freqs, amps)
