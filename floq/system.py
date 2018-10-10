"""
Contains the base `floq` class `System`.  This class should be exported into the
global `floq` namespace, so this module need not be accessed by end-users.
"""

import numpy as np
import abc
import logging
import functools
from . import evolution, types

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

@functools.singledispatch
def _canonicalise_operator(operator):
    # Base case handles iterables of `(mode, matrix)`.
    try:
        iterable = iter(operator)
        mode, hamiltonian = zip(*iterable)
        mode, hamiltonian = tuple(mode), tuple(hamiltonian)
    except [TypeError, ValueError]:
        msg = (f"Could not interpret type {type(argument)} as a"
               " Fourier-transformed matrix.  See the help for `hamiltonian`"
               " in `floq.System`.")
        raise TypeError(msg) from None
    for m in mode:
        if type(m) != int:
            raise TypeError(f"Invalid mode type {type(m)}.  Should be int.")
    for matrix in hamiltonian:
        failure = not isinstance(matrix, np.ndarray)\
                  or len(matrix.shape) is not 2\
                  or matrix.shape[0] != matrix.shape[1]
        if failure:
            msg = f"Matrix should be a 2D square numpy array, but is:\n{matrix}"
            raise ValueError(msg)
    return types.TransformedMatrix(mode, hamiltonian)

@_canonicalise_operator.register(np.ndarray)
def _canonicalise_ndarray(operator):
    if len(operator.shape) is not 3 or operator.shape[1] != operator.shape[2]:
        msg = (f"Invalid shape of operator {operator.shape}.  The shape must be"
               " `(mode, dimension, dimension)`, where the first index runs"
               " over the Fourier mode, and the second two indices define the"
               " square matrix operator.")
        raise ValueError(msg)
    mode_max = (operator.shape[0] - 1) // 2
    mode = tuple(range(-mode_max, mode_max + 1))
    return types.TransformedMatrix(mode, tuple(matrix for matrix in operator))

_canonicalise_operator.register(types.TransformedMatrix, lambda x: x)

@_canonicalise_operator.register(dict)
def _canonicalise_dict(dict_):
    # Pass through to the base case as a correct iterable.
    return _canonicalise_operator(dict_.items())

class System:
    """
    The base `floq` system class, providing methods to calculate the
    time-evolution operator and its derivatives for Fourier-transformed
    Hamiltonians.

    The base methods are `u()`, `du_dcontrols()` and `du_dt()`.  The convenience
    function `h_effective()` is also provided.


    Fourier_matrix_like types
    =========================

    All matrices (the Hamiltonian and each of its derivatives with respect
    to a control parameter) must be given in one of the allowable
    `Fourier_matrix_like` types, in Fourier-transformed form with respect to
    the principle frequency `frequency` (given in the initialiser).

    Types:
        iterable of (mode: int, matrix: 2D np.array of complex) --
            `mode` must be an `int`, which is the integer multiple of
            `frequency`.  This can be negative.  `matrix` is the matrix form of
            the operator at this Fourier mode, as a `numpy.array` of complex
            values.  This is the preferred input form (using any iterable
            container type).

        dict of (mode: int, matrix: 2D np.array of complex) --
            Everything is the same as the iterable form.

        3D np.array of complex --
            This is similar to the iterable form, except the modes in used are
            prescribed.  The shape of the array must be
                (n_modes, dimension, dimension)
            so `operator[i]` gives a 2D square matrix.  The modes are always
            interpreted as going
                -m, -m+1, ..., 0, 1, ..., m-1, m
            where `m = (n_modes - 1) // 2`.  This form is for backwards
            compatibiliity with the original implementation of `floq`.

        floq.types.TransformedMatrix --
            This is the canonical representation of a Fourier transformed
            matrix, which is used throughout the numerical evolution code.  This
            type is typically not intended to be instantiated or interacted with
            by the end-user.
    """
    def __init__(self, hamiltonian, dhamiltonian=None, n_zones=1, frequency=1.0,
                       sparse=True, decimals=8, cache=True):
        """
        Arguments --
        hamiltonian:
        | (*args, **kwargs) -> Fourier_matrix_like
        | Fourier_matrix_like --
            A function which takes any set of arguments and returns a Fourier
            representation of the matrix, or simply the matrix itself.  See the
            help of `floq.System` for details on `Fourier_matrix_like` types.

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
        | (*args, **kwargs) -> iterable of Fourier_matrix_like
        | iterable of Fourier_matrix_like
        | None --
            A function which takes any set of arguments and returns the
            derivatives of the Fourier representation of the matrix with respect
            to each of the control parameters in turn, or simply the resultant
            iterable itself if it is not parameter-dependent. The function
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
        self._args = None
        self._kwargs = None
        self._eigensystem = None
        self._n_components = None
        self._n_zones= n_zones
        self.cache = cache
        self.sparse = sparse
        self.decimals = decimals
        self.frequency = frequency
        self._hamiltonian_inner = _make_callable(hamiltonian)
        self._dhamiltonian_inner = _make_callable(dhamiltonian)

    @property
    def hamiltonian(self):
        return self._hamiltonian_inner
    @hamiltonian.setter
    def hamiltonian(self, value):
        self._hamiltonian_inner = _make_callable(hamiltonian)

    def _hamiltonian(self, *args, **kwargs):
        return _canonicalise_operator(self.hamiltonian(*args, **kwargs))

    @property
    def dhamiltonian(self):
        return self._dhamiltonian_inner
    @dhamiltonian.setter
    def dhamiltonian(self, value):
        self._dhamiltonian_inner = _make_callable(dhamiltonian)

    def _dhamiltonian(self, *args, **kwargs):
        dhamiltonian = self.dhamiltonian(*args, **kwargs)
        if dhamiltonian is None:
            return None
        return tuple(_canonicalise_operator(op) for op in dhamiltonian)

    @property
    def n_zones(self):
        return self._n_zones

    @n_zones.setter
    def n_zones(self, value):
        # Remove the known eigensystem to force recalculation on the next pass.
        self._eigensystem = None
        self._n_zones = value

    def _update_if_required(self, t: float, args, kwargs):
        """
        Update the underlying `FixedSystem` if the current version was created
        with a different set of control or time parameters.
        """
        if self._eigensystem is not None and self.cache\
           and _compare_args(self._args, args)\
           and _compare_kwargs(self._kwargs, kwargs):
            return
        hamiltonian = self._hamiltonian(*args, **kwargs)
        dhamiltonian = self._dhamiltonian(*args, **kwargs)
        min_n_zones = 2 * max((abs(x) for x in hamiltonian.mode)) + 1
        if self._n_zones is None or min_n_zones > self._n_zones:
            logging.debug(f"Increasing number of zones to {min_n_zones} to"
                          + " match the number of Fourier components in the"
                          + " Hamiltonian.")
            self.n_zones = min_n_zones
        self._eigensystem =\
            evolution.eigensystem(hamiltonian, dhamiltonian, self.n_zones,
                                  self.frequency, self.decimals, self.sparse)
        self._args = tuple(args)
        self._kwargs = kwargs.copy()

    def u(self, t: float, *args, **kwargs):
        """
        Calculate the time evolution operator of the stored Hamiltonian.
        """
        self._update_if_required(t, args, kwargs)
        return evolution.u(self._eigensystem, t)

    def du_dt(self, t: float, *args, **kwargs):
        """
        Calculate the derivative of the time-evolution operator with respect to
        time.
        """
        self._update_if_required(t, args, kwargs)
        return evolution.du_dt(self._eigensystem, t)

    def du_dcontrols(self, t: float, *args, **kwargs):
        """
        Calculate the derivatives of the time-evolution operator with respect to
        each of the control parameters in turn.
        """
        self._update_if_required(t, args, kwargs)
        return evolution.du_dcontrols(self._eigensystem, t)

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
