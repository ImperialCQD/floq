"""
Internal struct-style types for canonicalising data formats.  A
`collections.namedtuple` is understood by `numba`, which means we can pass
around grouped data without suffering a massive overhead when dealing with fast
numerics.

These needn't be exposed to the end user of the library, since they're just
internal types.
"""

import collections

TransformedMatrix = collections.namedtuple('TransformedMatrix',
                                           ('mode', 'matrix'))
TransformedMatrix.__doc__ =\
    """
    Internal representation of a Fourier-transformed Hamiltonian.  This is
    essentially a struct-of-arrays, with two related strands - an array of
    integers of the populated Fourier modes, and an array of the corresponding
    (2D complex) Hamiltonians.

    The idea is that the user will be able to specify their Hamiltonian in
    several different forms (since depending on the problem there are a few ways
    which make sense), but internally we will convert to a single canonical form
    which will then be used throughout.
    """
TransformedMatrix.mode.__doc__ =\
    """
    iterable of int

    An array of the populated modes of the Fourier transformed Hamiltonian.  The
    matrix corresponding to `mode[i]` should be stored in `hamiltonian[i]`.
    """
TransformedMatrix.matrix.__doc__ =\
    """
    indexable of 2D np.array of complex

    The Hamiltonian matrices for each populated Fourier mode.  The Hamiltonian
    in `matrix[i]` should have Fourier mode `mode[i]`.
    """


Eigensystem = collections.namedtuple('Eigensystem', (
                                         'frequency',
                                         'quasienergies',
                                         'k_eigenvectors',
                                         'initial_floquet_bras',
                                         'abstract_ket_coefficients',
                                         'k_derivatives',
                                    ))
Eigensystem.__doc__ =\
    """
    Struct type for holding the time-independent parameters related to the
    diagonalised Floquet matrix.  This is simply an internal type, and should
    not need to be exposed to the user.  This allows for many objects to be
    passed simultaneously in a `numba`-compatible manner.
    """
Eigensystem.frequency.__doc__ = "The principle frequency of the Floquet matrix."
Eigensystem.quasienergies.__doc__ = "The eigenvalues of the Floquet matrix."
Eigensystem.k_eigenvectors.__doc__ = "The eigenvectors of the Floquet matrix."
Eigensystem.initial_floquet_bras.__doc__ =\
    "The conjugate Floquet basis vectors at the initial time."
Eigensystem.abstract_ket_coefficients.__doc__ =\
    "The time-independent part of the abstract frequency kets."
Eigensystem.k_derivatives.__doc__ =\
    "Matrix representation of the derivatives of the Floquet matrix."


# I use this custom sparse column representation of a matrix for compatibility
# with `numba`, since it can't understand `scipy.sparse` matrices.  I couple
# this with a simple implementation of a left dot product (i.e. vector . matrix)
# to get a speed up on the heaviest part of `evolution.du_dcontrols()`.
ColumnSparseMatrix = collections.namedtuple('ColumnSparseMatrix',
                                            ('in_column', 'row', 'value'))
ColumnSparseMatrix.__doc__ =\
    """
    A custom sparse matrix format for storing the derivatives of the Floquet
    matrix, which is `numba`-compatible.  This makes operations of the form
    `vector . matrix` efficient, which is necessary in the calculation of the
    derivatives of the time-evolution operator.

    The structure is such that `in_column` is a count of how many explicit
    values are given in each column.  For each value, these are then entered in
    a top-to-bottom, left-to-right order into the other two arrays in this type
    (`row` and `value`).

    For example, the matrix:
        4   0   1
        0   1   0
        0   0   3
    would become
        in_column = np.array([1, 1, 2])
              row = np.array([0, 1, 0, 2])
            value = np.array([4, 1, 1, 3])

    It is difficult to determine the value of an arbitrary index of the matrix,
    because it is tricky to find which column a particular row index and value
    correspond to without having evaluated all the non-zero elements before the
    requested one.  This does not limit the speed of `vector . matrix`
    operations, however.
    """
ColumnSparseMatrix.in_column.__doc__ =\
    """
    np.array of int >= 0, shape=(matrix_dimension,)

    `in_column[i]` is the number of explicitly given (i.e. non-zero) elements in
    column `i` of the full matrix.
    """
ColumnSparseMatrix.row.__doc__ =\
    """
    np.array(dtype=np.int, shape=(np.sum(in_column)))

    The row index for the explicitly given value.  These should be sorted by
    column index, but the sorting within any particular column is arbitrary
    except that `row[i]` should be the row index for `value[i]`.
    """
ColumnSparseMatrix.value.__doc__ =\
    """
    np.array(dtype=np.complex128, shape=(np.sum(in_column)))

    The explicitly given value which matches the column and row index at this
    position.
    """
