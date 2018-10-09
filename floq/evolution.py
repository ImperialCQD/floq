"""
Contains all the heavy-lifting numerical apparatus for doing Floquet-theory
calculations.  This need not be called by the end-user, since everything happens
behind the scenes in the `System` class.

There are largely two categories of functions in this module - ones which are
involved in the creation and diagonalisation of the Floquet matrix, and ones
which take a diagonalised matrix and return objects related to the
time-evolution operator.  The type `types.Eigensystem` is the link between these
two - this contains all the information necessary to build all the other
properties after a diagonalisation, and is cached by the `System` class.
"""

import numba
import numpy as np
import scipy.sparse.linalg
import logging
from . import linalg, types

_log = logging.getLogger(__name__)

def eigensystem(hamiltonian, dhamiltonian, n_zones, frequency, decimals=8,
                sparse=True):
    """
    Calculate the time-invariant eigensystem of the Floquet system.  This needs
    to be recalculated whenever the Hamiltonian (or its derivatives) change, but
    not if the time changes.  The result of this can be passed to the other
    calculation routines.

    If the Hamiltonian's derivative is given as `None`, then the derivatives of
    the `K` matrix won't be build, which saves time but prevents the usage of
    the `du_dcontrols()` function.

    Arguments --
    hamiltonian: 3D np.array of complex --
        This must be the matrix of a Hamiltonian, split into Fourier components
        in the same manner used in the return values of
        `floq.System.hamiltonian()`.

    dhamiltonian: 4D np.array of complex | None --
        Optionally, the matrix form of the derivatives of a Hamiltonian, as in
        the output of `floq.System.dhamiltonian()`.  If not supplied, then the
        resulting `Eigensystem` cannot be used with the `du_dcontrols()`
        function.

    n_zones: odd int --
        The number of Brillouin zones to use when creating the Floquet matrix.

    frequency: float --
        The frequency with which the Hamiltonian is periodic.

    decimals: int --
        The number of decimal places of precision to use when comparing floats
        for equality.

    sparse: bool -- Whether to use sparse matrix algebra.

    Returns:
    Eigensystem --
        A collection of parameters that are not time-dependent, which can be
        passed to the time-specific functions.
    """
    n_components, dimension = hamiltonian.shape[0:2]
    k = assemble_k_sparse(hamiltonian, n_zones, frequency) if sparse\
        else assemble_k(hamiltonian, n_zones, frequency)
    k_derivatives = None if dhamiltonian is None\
                    else assemble_dk(dhamiltonian, n_zones)
    quasienergies, k_eigenvectors = diagonalise(k, dimension, frequency,
                                                decimals)
    # Sum the eigenvectors along the Fourier-mode axis at `time = 0` to contract
    # the abstract Hilbert space back to the original one.
    initial_floquet_bras = np.conj(np.sum(k_eigenvectors, axis=1))
    fourier_modes = np.arange((1-n_zones)//2, 1 + (n_zones//2))
    abstract_ket_coefficients = 1j * frequency * fourier_modes
    return types.Eigensystem(frequency, quasienergies, k_eigenvectors,
                             initial_floquet_bras, abstract_ket_coefficients,
                             k_derivatives)

@numba.njit
def current_floquet_kets(eigensystem, time):
    """
    Get the Floquet basis kets at a given time.  These are the
        |psi_j(t)> = exp(-i energy[j] t) |phi_j(t)>,
    using the notation in Marcel's thesis, equation (1.13).
    """
    weights = np.exp(time * eigensystem.abstract_ket_coefficients)
    weights = weights.reshape((1, -1, 1))
    return np.sum(weights * eigensystem.k_eigenvectors, axis=1)

@numba.njit
def d_current_floquet_kets(eigensystem, time):
    """
    Get the time derivatives of the Floquet basis kets
        (d/dt)|psi_j(t)> = -i energy[j] exp(-i energy[j] t) |phi_j(t)>
                           + exp(-i energy[j] t) (d/dt)|phi_j(t)>,
    which are used in calculating `dU/dt`.
    """
    weights = np.exp(time * eigensystem.abstract_ket_coefficients)
    weights = weights * eigensystem.abstract_ket_coefficients
    weights = weights.reshape((1, -1, 1))
    return np.sum(weights * eigensystem.k_eigenvectors, axis=1)

@numba.njit
def u(eigensystem, time):
    """
    Calculate the time-evolution operator at a certain time, using a
    pre-computed eigensystem.
    """
    dimension = eigensystem.quasienergies.shape[0]
    out = np.zeros((dimension, dimension), dtype=np.complex128)
    kets = current_floquet_kets(eigensystem, time)
    energy_phases = np.exp(-1j * time * eigensystem.quasienergies)
    # This sum _can_ be achieved using only vectorised numpy operations, but it
    # involves allocating space for the whole set of outer products.  Easier to
    # just use numba to compile the loop.
    for mode in range(dimension):
        out += np.outer(energy_phases[mode] * kets[mode],
                        eigensystem.initial_floquet_bras[mode])
    return out

@numba.njit
def du_dt(eigensystem, time):
    """
    Calculate the time derivative of a time-evolution operator at a certain
    time, using a pre-computed eigensystem.
    """
    dimension = eigensystem.quasienergies.shape[0]
    out = np.zeros((dimension, dimension), dtype=np.complex128)
    # Manually allocate calculation space for inside the loop.
    tmp = np.zeros_like(out)
    # These two function calls have some redundancy, but it's not really
    # important in the scheme of things.
    kets = current_floquet_kets(eigensystem, time)
    dkets_dt = d_current_floquet_kets(eigensystem, time)
    energy_factors = -1j * eigensystem.quasienergies
    energy_phases = np.exp(time * energy_factors)
    for mode in range(dimension):
        # Use the pre-allocated computation space to save allocations.
        np.outer(energy_phases[mode] * dkets_dt[mode],
                 eigensystem.initial_floquet_bras[mode], out=tmp)
        out += tmp
        np.outer(energy_phases[mode] * kets[mode],
                 eigensystem.initial_floquet_bras[mode], out=tmp)
        out += energy_factors[mode] * tmp
    return out

@numba.njit
def conjugate_rotate_into(out, input, amount):
    """
    Equivalent to `out = np.conj(np.roll(input, amount, axis=0))`, but `roll()`
    isn't supported by numba.  Also, we can directly write into `out` so we
    don't have any allocations in tight loops.
    """
    amount = amount % input.shape[0]
    if amount < 0:
        out[:amount] = np.conj(input[abs(amount):])
        out[amount:] = np.conj(input[:abs(amount)])
    elif amount > 0:
        out[amount:] = np.conj(input[:-amount])
        out[:amount] = np.conj(input[-amount:])
    else:
        out[:] = np.conj(input)
@numba.njit
def _column_sparse_ldot(vector, matrix):
    out = np.zeros_like(vector)
    i = 0
    for column, count in enumerate(matrix.in_column):
        for _ in range(count):
            out[column] += vector[matrix.row[i]] * matrix.value[i]
            i += 1
    return out

@numba.njit
def integral_factors(eigensystem, time):
    """
    Calculate the "integral factors" for use in the control-derivatives of the
    time-evolution operator.  These are the
        e(j, j'; delta mu)
    from equation (1.48) in Marcel's thesis.
    """
    n_zones = eigensystem.k_eigenvectors.shape[1]
    dimension = eigensystem.k_eigenvectors.shape[2]
    energies = eigensystem.quasienergies
    frequency = eigensystem.frequency
    energy_phases = np.exp(-1j * time * energies)
    differences = np.arange(1.0 - n_zones, n_zones)
    diff_exponentials = np.exp(1j * time * frequency * differences)
    out = np.empty((differences.shape[0], dimension, dimension),
                   dtype=np.complex128)
    for diff_index in range(differences.shape[0]):
        separation = frequency * differences[diff_index]
        exponential = diff_exponentials[diff_index]
        for i in range(energies.shape[0]):
            for j in range(energies.shape[0]):
                prefactor = energy_phases[j] * exponential
                denom = energies[i] - energies[j] + separation
                if denom == 0.0:
                    out[diff_index, i, j] = -1j * time * prefactor
                else:
                    numer = energy_phases[i] - prefactor
                    out[diff_index, i, j] = numer / denom
    return out

@numba.njit
def combined_factors(eigensystem, time):
    """
    Calculate the "combined factors" for use in the control-derivatives of the
    time evolution operator.  These are the
        f(j, j'; delta mu)
    from equations (1.50) and (2.7) in Marcel's thesis.
    """
    n_parameters = len(eigensystem.k_derivatives)
    n_zones = eigensystem.k_eigenvectors.shape[1]
    dimension = eigensystem.k_eigenvectors.shape[2]
    factors = np.empty((2*n_zones - 1, dimension, dimension, n_parameters),
                       dtype=np.complex128)
    rolled_k_eigenbra = np.zeros((n_zones, dimension),
                                 dtype=np.complex128)
    expectation_left = np.empty((n_parameters, n_zones * dimension),
                                dtype=np.complex128)
    k_eigenkets = eigensystem.k_eigenvectors
    energies = eigensystem.quasienergies
    integral_terms = integral_factors(eigensystem, time)
    for diff_index, diff in enumerate(range(1 - n_zones, n_zones)):
        for i in range(dimension):
            conjugate_rotate_into(rolled_k_eigenbra, k_eigenkets[i], diff)
            bra = rolled_k_eigenbra.ravel()
            for p in range(n_parameters):
                expectation_left[p] =\
                    _column_sparse_ldot(bra, eigensystem.k_derivatives[p])
            for j in range(dimension):
                for parameter in range(n_parameters):
                    expectation = expectation_left[parameter]\
                                  @ k_eigenkets[j].ravel()
                    factors[diff_index, i, j, parameter] =\
                        integral_terms[diff_index, i, j] * expectation
    return factors

@numba.njit
def du_dcontrols(eigensystem, time):
    """
    Calculate the derivatives of time-evolution operator with respect to the
    control parameters of the Hamiltonian at a certain time, using a
    pre-computed eigensystem.  This is only possible if the eigensystem was
    created using the Hamiltonian derivatives as well.
    """
    n_parameters = len(eigensystem.k_derivatives)
    n_zones, dimension = eigensystem.k_eigenvectors.shape[1:3]
    out = np.zeros((n_parameters, dimension, dimension), dtype=np.complex128)
    if n_parameters == 0:
        return out
    factors = combined_factors(eigensystem, time)
    current_kets = current_floquet_kets(eigensystem, time)
    k_eigenbras = np.conj(eigensystem.k_eigenvectors)
    for i in range(dimension):
        for j in range(dimension):
            bra = np.zeros((n_parameters, dimension), dtype=np.complex128)
            for zone_i in range(n_zones):
                factor = np.sum(factors[zone_i : zone_i+n_zones, i, j], axis=0)
                bra += factor.reshape(-1, 1) * k_eigenbras[j, zone_i]
            for parameter in range(n_parameters):
                out[parameter] += np.outer(current_kets[i], bra[parameter])
    return out

@numba.njit
def _k_ijv_constructor(hamiltonian, n_zones, frequency):
    """
    Returns a tuple of
        values, (row_indices, col_indices)
    where all three names are 1D `numpy` arrays.  These determine the `K` matrix
    in 'ijv' or 'triplet' format. See
        http://www.scipy-lectures.org/scipy_sparse/coo_matrix.html
    and other associated `scipy` pages for more information.  This triplet form
    can be passed to the `coo`, `csc` or `csr` sparse matrix constructors,
    resulting in an efficient creation.
    """
    nonzeros = [hamiltonian[i].nonzero() for i in range(hamiltonian.shape[0])]
    rows, cols = [x[0] for x in nonzeros], [x[1] for x in nonzeros]
    dimension = hamiltonian.shape[1]
    mid = (len(rows) - 1) // 2
    n_elements = dimension * n_zones # include extra space for diagonal
    for i, row in enumerate(rows):
        n_elements += row.size * (n_zones - abs(mid - i))
    row_out = np.empty(n_elements, dtype=np.int64)
    col_out = np.empty_like(row_out)
    val_out = np.empty(n_elements, dtype=np.complex128)
    start = 0
    for i in range(len(rows)):
        row, col = rows[i], cols[i]
        val = np.array([hamiltonian[i,row[k],col[k]] for k in range(row.size)])
        start_row, start_col = max(0, i - mid), max(0, mid - i)
        n_blocks = n_zones - abs(mid - i)
        for j in range(n_blocks):
            row_out[start : start+row.size] = row + (start_row+j)*dimension
            col_out[start : start+row.size] = col + (start_col+j)*dimension
            val_out[start : start+row.size] = val
            start += row.size
    # When converting a `coo`-style matrix to `csc` or `csr`, duplicated
    # coordinates have their values summed.  I add the diagonal `1, n*frequency`
    # operator values here as completely separated entries in the sparse matrix,
    # because it's easier to reason about what's happening.  I then use the
    # documented summation property on conversion to `csc`.  See
    # docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    indices = np.arange(n_zones * dimension)
    row_out[start:] = indices
    col_out[start:] = indices
    for j in range(n_zones):
        val_out[start : start + dimension] = (j - n_zones//2) * frequency
        start += dimension
    return val_out, (row_out, col_out)

def assemble_k_sparse(hamiltonian, n_zones, frequency):
    """
    Directly assemble `K` as a sparse matrix in `csc` (Compressed Sparse Column)
    format.  The sparser `K` is, the more efficient this way of doing things is.
    """
    elements = _k_ijv_constructor(hamiltonian, n_zones, frequency)
    size = n_zones * hamiltonian.shape[1]
    # Use `csc` format for efficiency in the diagonalisation routine.
    return scipy.sparse.csc_matrix(elements, shape=(size, size))

@numba.njit
def _n_to_i(num, n):
    """Translate num, ranging from -(n-1)/2 through (n-1)/2 into an index i from
    0 to n-1.  If num > (n-1)/2, map it into the interval.  This is necessary to
    translate from a physical Fourier mode number to an index in an array."""
    return (num + (n - 1) // 2) % n

@numba.njit
def _i_to_n(i, n):
    """Translate index i, ranging from 0 to n-1 into a number from -(n-1)/2
    through (n-1)/2.  This is necessary to translate from an index to a physical
    Fourier mode number."""
    return i - (n - 1) // 2

@numba.njit
def _set_block(block, matrix, dim_block, n_block, row, col):
    start_row = row * dim_block
    start_col = col * dim_block
    stop_row = start_row + dim_block
    stop_col = start_col + dim_block
    matrix[start_row:stop_row, start_col:stop_col] = block

@numba.njit
def assemble_k(hf, nz, omega):
    nc, dim = hf.shape[0:2]
    k_dim = dim * nz
    hf_max = (nc - 1) // 2
    k = np.zeros((k_dim, k_dim), dtype=np.complex128)
    for n in range(-hf_max, hf_max + 1):
        current = hf[_n_to_i(n, nc)]
        row = max(0, n)  # if n < 0, start at row 0
        col = max(0, -n)  # if n > 0, start at col 0
        stop_row = min(nz - 1 + n, nz - 1)
        stop_col = min(nz - 1 - n, nz - 1)
        while row <= stop_row and col <= stop_col:
            if n == 0:
                block = current + np.eye(dim) * omega * _i_to_n(row, nz)
                _set_block(block, k, dim, nz, row, col)
            else:
                _set_block(current, k, dim, nz, row, col)
            row = row + 1
            col = col + 1
    return k

@numba.njit
def _dense_to_sparse(matrix):
    """
    Convert a dense 2D numpy array of complex into the custom
    `ColumnSparseMatrix` format.  This is probably not the most efficient way of
    doing things, but it's simple and easy to read.
    """
    in_column = np.zeros(matrix.shape[1], dtype=np.int64)
    col, row = matrix.T.nonzero()
    value = np.empty(col.size, dtype=np.complex128)
    for i in range(col.size):
        in_column[col[i]] += 1
        value[i] = matrix[row[i], col[i]]
    return types.ColumnSparseMatrix(in_column, row, value)

@numba.njit
def _single_dk_sparse(dhamiltonian, n_zones):
    """
    Create a single sparse matrix for a single derivative.
    """
    n_modes, dimension = dhamiltonian.shape[0:2]
    modes = [_dense_to_sparse(dhamiltonian[i]) for i in range(n_modes)]
    mode_mid = (n_modes - 1) // 2
    n_elements = 0
    for i, mode in enumerate(modes):
        n_elements += mode.value.size * (n_zones - abs(mode_mid - i))
    in_column = np.zeros(n_zones * dimension, dtype=np.int64)
    row = np.empty(n_elements, dtype=np.int64)
    value = np.empty(n_elements, dtype=np.complex128)
    main_ptr = 0
    block_ptr = np.zeros(n_modes, dtype=np.int64)
    block_mid_row = 0
    for zone_column in range(n_zones):
        for block_column in range(dimension):
            start_mode = max(0, mode_mid - zone_column)
            end_mode = min(n_modes, mode_mid + n_zones - zone_column)
            for j in range(start_mode, end_mode):
                n_to_add = modes[j].in_column[block_column]
                in_column[zone_column*dimension + block_column] += n_to_add
                row_add = (block_mid_row + j - mode_mid) * dimension
                for _ in range(n_to_add):
                    row[main_ptr] = modes[j].row[block_ptr[j]] + row_add
                    value[main_ptr] = modes[j].value[block_ptr[j]]
                    main_ptr += 1
                    block_ptr[j] += 1
        block_ptr[:] = 0
        block_mid_row += 1
    return types.ColumnSparseMatrix(in_column, row, value)

@numba.njit
def assemble_dk(dhamiltonians, n_zones):
    """
    Creates the `dK` matrix as a list of the custom `ColumnSparseMatrix` tuple.
    The only operation we need with the output matrix is an inner product `<x|M`
    so the column sparse format is very efficient.

    I use this poor-man's sparse matrix beacuse `numba` doesn't know about
    `scipy.sparse` matrices.
    """
    return [_single_dk_sparse(dhamiltonians[i], n_zones)
            for i in range(dhamiltonians.shape[0])]

def first_brillouin_zone(eigenvalues, eigenvectors, n_values, edge):
    """
    Return the `n_values` eigenvalues (and corresponding eigenvectors) which
    fall within the first "Brillioun zone" whos edge is `edge`.  This function
    takes care to select values from only one edge of the zone, and raises a
    `RuntimeError` if it cannot safely do so.

    The inputs `eigenvalues` and `edge` must be rounded to the desired
    precision for this function.

    Arguments:
    eigenvalues: 1D np.array of float --
        The eigenvalues to choose from.  This should be rounded to the desired
        precision (because the floating-point values will be compared directly
        to the edge value).

    eigenvectors: 2D np.array of complex --
        The eigenvectors corresponding to the eigenvalues.  The first index runs
        over the number of vectors, so `eigenvalues[i]` is the eigenvalue
        corresponding to the eigenvector `eigenvectors[i]`.  Note: this is the
        transpose of the return value of `np.linalg.eig` and family.

    n_values: int -- The number of eigenvalues to find.

    edge: float --
        The edge of the first Brillioun zone.  This should be rounded to the
        desired precision.

    Returns:
    eigenvalues: 1D np.array of float -- The selected eigenvalues (sorted).
    eigenvectors: 2D np.array of complex --
        The eigenvectors corresponding to the selected eigenvalues.  The first
        index corresponds to the index of the `eigenvalues` output.
    """
    order = eigenvalues.argsort()
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[order]
    lower = np.searchsorted(eigenvalues, -edge, side='left')
    upper = np.searchsorted(eigenvalues, edge, side='right')
    n_lower_edge = n_upper_edge = 0
    while eigenvalues[lower + n_lower_edge] == -edge:
        n_lower_edge += 1
    # Additional `-1` because `searchsorted(side='right')` gives us the index
    # after the found element.
    while eigenvalues[upper - n_upper_edge - 1] == edge:
        n_upper_edge += 1
    n_not_on_edge = (upper - n_upper_edge) - (lower + n_lower_edge)
    log_message = " ".join([
        f"Needed {n_values} eigenvalues in the first zone.",
        f"Found {n_lower_edge}, {n_not_on_edge}, {n_upper_edge} on the",
        "lower edge, centre zone, upper edge respectively.",
    ])
    _log.debug(log_message)
    if n_not_on_edge == n_values:
        lower, upper = lower + n_lower_edge, upper - n_upper_edge
    elif n_not_on_edge + n_lower_edge == n_values:
        lower, upper = lower, upper - n_upper_edge
    elif n_not_on_edge + n_upper_edge == n_values:
        lower, upper = lower + n_lower_edge, upper
    else:
        exception_message = " ".join([
            "Could not resolve the first Brillouin zone safely.",
            "You could try increasing the tolerance (decreasing the 'decimals'",
            "field), or adding a small constant term to your Hamiltonian.",
        ])
        raise RuntimeError(exception_message)
    return eigenvalues[lower:upper], eigenvectors[lower:upper]

def diagonalise(k, h_dimension, frequency, decimals):
    """
    Find the eigenvalues and eigenvectors of the Floquet matrix `k`
    corresponding to the first "Brillioun zone".  The eigenvectors corresponding
    to degenerate eigenvalues are orthogonalised, where degeneracy and
    orthoganlisation are done to a precision defined by `decimals`.

    Arguments --
    k: 2D np.array of complex | scipy.sparse.spmatrix --
        The full matrix form of the Floquet matrix.  This can be given as either
        a dense `numpy` array, or any form of `scipy.sparse` array.  In the
        latter case, the eigenvalues and vectors are found iteratively with a
        shift-invert method, which can cause issues when eigenvalues fall
        exactly on zero, or exactly on the border.  The former case can
        typically be solved by adding a term proportional to the identity onto
        the Hamiltonian, and the latter should largely be taken care of by the
        code.

    h_dimension: int -- The dimension of the Hamiltonian.

    frequency: float --
        The angular frequency with which the Hamiltonian is periodic.

    decimals: int --
        The number of decimal places to use as a precision for orthogonalisation
        and comparison of degenerate eigenvalues.

    Returns --
    eigenvalues: 1D np.array of float --
        The eigenvalues of the `k` matrix which fall within the first Brillouin
        zone.
    eigenvectors: np.array(dtype=np.complex128,
                           shape=(h_dimension, n_zones, h_dimension)) --
        The eigenvectors corresponding to the chosen eigenvalues.  The first
        index matches the index of `eigenvalues`, then each eigenvector is
        shaped into the Brillouin zone blocks, so the second index runs along
        Floquet kets corresponding to a certain (potentially degenerate)
        quasi-energy.
    """
    if scipy.sparse.issparse(k):
        # We get twice as many eigenvalues as necessary so we can guarantee we
        # have a full set in the first Brillouin zone, even if all of them fall
        # on the very edge of the zone.  If this were to happen and we were only
        # taking the absolutely necessary number of eigenvalues, we would
        # sometimes duplicate an eigenvector without intending to.
        eigenvalues, eigenvectors =\
            scipy.sparse.linalg.eigs(k, k=2*h_dimension, sigma=0.0)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(k)
    eigenvalues = np.round(np.real(eigenvalues), decimals=decimals)
    eigenvectors = np.transpose(eigenvectors)
    edge = np.round(0.5 * frequency, decimals=decimals)
    eigenvalues, eigenvectors = first_brillouin_zone(eigenvalues, eigenvectors,
                                                     h_dimension, edge)
    for degeneracy in find_duplicates(eigenvalues):
        eigenvectors[degeneracy] = linalg.gram_schmidt(eigenvectors[degeneracy])
    n_zones = k.shape[0] // h_dimension
    return eigenvalues, eigenvectors.reshape(h_dimension, n_zones, h_dimension)

def find_duplicates(array):
    """
    Given a sorted 1D array of values, return an iterator where each element
    corresponds to one degenerate eigenvalue, and the element is a list of
    indices where that eigenvalue occurs.
    For example,
        find_duplicates([0, 0, 0, 1, 2, 2, 3])
    returns
        [[0, 1, 2], [4, 5]].
    Arguments --
    array: 1D sorted np.array --
        The array to find duplicates in.  Should be rounded to the required
        precision already.
    Returns --
    duplicate_sets: iterator of np.array of int --
        An iterator yielding arrays of the indices of each duplicate entry.
        Entries which are not duplicated will not be referenced in the output.
    """
    indices = np.arange(array.shape[0])
    _, start_indices = np.unique(array, return_index=True)
    # start_indices will always contain 0 first, but np.split doesn't need it.
    return filter(lambda x: x.size > 1, np.split(indices, start_indices[1:]))
