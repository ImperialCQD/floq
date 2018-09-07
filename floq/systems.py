import numpy as np
import abc
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
        hamiltonian:
        | (*args, **kwargs) -> 3D np.array of complex,
        | 3D np.array of complex --
            A function which takes any set of arguments and returns a Fourier
            representation of the matrix, or simply the matrix itself.  The
            shape of the output array is
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

            The function can take any number of positional and keyword
            arguemnts.  These are supplied by passing additional positional and
            keyword arguments to `self.u()`, `self.du_dcontrols()` and the other
            instance methods.  After the necessary values for those functions
            are filled, any additional parameters will be passed on to
            `hamiltonian`.  This typically means that `self.u()` is called as
            `self.u(t, controls)` or something similar, and the signature of
            `hamiltonian` is `hamiltonian(controls) -> np.array`.

            When caching, the arguments of `hamiltonian` are only
            shallow-copied to prevent limitations of the types of arguments that
            can be passed (deep-copy would make the corresponding equality check
            very difficult in the general case).  This means that caching may
            erroneously activate if the arguments are mutated in-place.
            Do not mutate arguments if this is a concern - only pass new
            instances (or disable caching, though this isn't recommended if you
            care about both `u()` and `du_d*()`).

        dhamiltonian:
        | (*args, **kwargs) -> 4D np.array of complex
        | 4d np.array of complex --
            A function which takes any set of arguments and returns the
            derivatives of the Fourier representation of the matrix with respect
            to each of the control parameters in turn, or simply the matrix
            itself.  The shape is
                (ncontrols, modes, N, N),
            so the first index runs over the controls, and the remaining indices
            are the same as the output of `hamiltonian()`.  The function
            arguments must match those of `hamiltonian`.

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
        self.hamiltonian = self.__make_callable(hamiltonian)
        self.dhamiltonian = self.__make_callable(dhamiltonian)

    def __make_callable(self, maybe_callable):
        return maybe_callable if hasattr(maybe_callable, '__call__')\
               else lambda *args, **kwargs: maybe_callable

    def __compare_args(self, one, two):
        if len(one) != len(two):
            return False
        for a, b in zip(one, two):
            if not np.array_equal(a, b):
                return False
        return True

    def __compare_kwargs(self, one, two):
        items = set(one)
        if items != set(two):
            return False
        for item in items:
            if not np.array_equal(one[item], two[item]):
                return False
        return True

    def __update_if_required(self, t: float, args, kwargs):
        """
        Update the underlying `FixedSystem` if the current version was created
        with a different set of control or time parameters.
        """
        if self.cache\
           and self.__t == t\
           and self.__compare_args(self.__args, args)\
           and self.__compare_kwargs(self.__kwargs, kwargs):
            return
        hamiltonian = self.hamiltonian(*args, **kwargs)
        dhamiltonian = self.dhamiltonian(*args, **kwargs)
        self.__fixed = core.FixedSystem(hamiltonian, dhamiltonian, self.nz,
                                        self.omega, t,
                                        decimals=self.decimals,
                                        sparse=self.sparse,
                                        max_nz=self.max_nz)
        self.__t = t
        self.__args = tuple(args)
        self.__kwargs = kwargs.copy()

    def u(self, t: float, *args, **kwargs):
        """
        Calculate the time evolution operator of the stored Hamiltonian.
        """
        self.__update_if_required(t, args, kwargs)
        return self.__fixed.u

    def du_dt(self, t: float, *args, **kwargs):
        """
        Calculate the derivative of the time-evolution operator with respect to
        time.
        """
        self.__update_if_required(t, args, kwargs)
        return self.__fixed.du_dt

    def du_dcontrols(self, t: float, *args, **kwargs):
        """
        Calculate the derivatives of the time-evolution operator with respect to
        each of the control parameters in turn.
        """
        self.__update_if_required(t, args, kwargs)
        return self.__fixed.du_dcontrols

    def h_effective(self, t: float, *args, **kwargs):
        u = self.u(t, *args, **kwargs)
        du_dt = self.du_dt(t, *args, **kwargs)
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
    _hamiltonian = np.zeros((3, 2, 2), dtype=np.complex128)
    _hamiltonian[1, 0, 0] = up_energy
    _hamiltonian[1, 1, 1] = down_energy
    def hamiltonian(controls):
        _hamiltonian[0, 1, 0] = _hamiltonian[2, 0, 1] = controls[0]
        return _hamiltonian
    dhamiltonian = np.zeros((1, 3, 2, 2), dtype=np.complex128)
    dhamiltonian[0, 0, 1, 0] = dhamiltonian[0, 2, 0, 1] = 1.0
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
    dhamiltonian = _spin_d_hamiltonian(n_components)
    return System(hamiltonian, dhamiltonian, **kwargs)

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
