import numpy as np
import abc
import logging
from . import evolution

def _make_callable(maybe_callable):
    return maybe_callable if hasattr(maybe_callable, '__call__')\
           else lambda *args, **kwargs: maybe_callable

def _compare_args(one, two):
    if len(one) != len(two):
        return False
    for a, b in zip(one, two):
        if not np.array_equal(a, b):
            return False
    return True

def _compare_kwargs(one, two):
    items = set(one)
    if items != set(two):
        return False
    for item in items:
        if not np.array_equal(one[item], two[item]):
            return False
    return True

class System:
    """
    The base `floq` system class, providing methods to calculate the
    time-evolution operator and its derivatives for Fourier-transformed
    Hamiltonians.

    The base methods are `u()`, `du_dcontrols()` and `du_dt()`.  The convenience
    function `h_effective()` is also provided.
    """
    def __init__(self, hamiltonian, dhamiltonian=None, n_zones=1, frequency=1.0,
                       sparse=True, decimals=8, cache=True):
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

        n_zones: odd int > 0 --
            The number of 'Brillouin zones' to be considered.

        frequency: float > 0 --
            The numerical value of the principle frequency (the frequency that
            the Fourier transform is with respect to).

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
        self.__args = None
        self.__kwargs = None
        self.__eigensystem = None
        self.__n_components = None
        self.__n_zones= n_zones
        self.cache = cache
        self.sparse = sparse
        self.decimals = decimals
        self.frequency = frequency
        self.hamiltonian = _make_callable(hamiltonian)
        self.dhamiltonian = _make_callable(dhamiltonian)

    @property
    def n_zones(self):
        return self.__n_zones

    @n_zones.setter
    def n_zones(self, value):
        # Remove the known eigensystem to force recalculation on the next pass.
        self.__eigensystem = None
        self.__n_zones = value

    def __update_if_required(self, t: float, args, kwargs):
        """
        Update the underlying `FixedSystem` if the current version was created
        with a different set of control or time parameters.
        """
        if self.__eigensystem is not None and self.cache\
           and _compare_args(self.__args, args)\
           and _compare_kwargs(self.__kwargs, kwargs):
            return
        hamiltonian = self.hamiltonian(*args, **kwargs)
        dhamiltonian = self.dhamiltonian(*args, **kwargs)
        n_components = hamiltonian.shape[0]
        if self.__n_zones is None or n_components > self.__n_zones:
            logging.debug(f"Increasing number of zones to {n_components} to"
                          + " match the number of Fourier components in the"
                          + " Hamiltonian.")
            self.n_zones = n_components
        self.__eigensystem =\
            evolution.eigensystem(hamiltonian, dhamiltonian, self.n_zones,
                                  self.frequency, self.decimals, self.sparse)
        self.__args = tuple(args)
        self.__kwargs = kwargs.copy()

    def u(self, t: float, *args, **kwargs):
        """
        Calculate the time evolution operator of the stored Hamiltonian.
        """
        self.__update_if_required(t, args, kwargs)
        return evolution.u(self.__eigensystem, t)

    def du_dt(self, t: float, *args, **kwargs):
        """
        Calculate the derivative of the time-evolution operator with respect to
        time.
        """
        self.__update_if_required(t, args, kwargs)
        return evolution.du_dt(self.__eigensystem, t)

    def du_dcontrols(self, t: float, *args, **kwargs):
        """
        Calculate the derivatives of the time-evolution operator with respect to
        each of the control parameters in turn.
        """
        self.__update_if_required(t, args, kwargs)
        return evolution.du_dcontrols(self.__eigensystem, t)

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
