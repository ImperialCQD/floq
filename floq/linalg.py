import numba
import numpy as np

@numba.jit(nopython=True)
def n_to_i(num, n):
    """Translate num, ranging from -(n-1)/2 through (n-1)/2 into an index i from
    0 to n-1.  If num > (n-1)/2, map it into the interval.  This is necessary to
    translate from a physical Fourier mode number to an index in an array."""
    return (num + (n - 1) // 2) % n

@numba.jit(nopython=True)
def i_to_n(i, n):
    """Translate index i, ranging from 0 to n-1 into a number from -(n-1)/2
    through (n-1)/2.  This is necessary to translate from an index to a physical
    Fourier mode number."""
    return i - (n - 1) // 2

@numba.jit(nopython=True)
def get_block(matrix, dim_block, n_block, row, col):
    start_row = row * dim_block
    start_col = col * dim_block
    stop_row = start_row + dim_block
    stop_col = start_col + dim_block
    return matrix[start_row:stop_row, start_col:stop_col]

@numba.jit(nopython=True)
def set_block(block, matrix, dim_block, n_block, row, col):
    start_row = row * dim_block
    start_col = col * dim_block
    stop_row = start_row + dim_block
    stop_col = start_col + dim_block
    matrix[start_row:stop_row, start_col:stop_col] = block

def is_unitary(u, digits):
    """Return True if u^dagger u is equal to the unit matrix with the given
    tolerance."""
    unitary = np.eye(u.shape[0], dtype=np.complex128)
    # The rounding is required in edge cases?
    product = np.round(np.conj(u.T) @ u, digits - 1)
    return np.allclose(product, unitary, atol=0.1**digits)

def gram_schmidt(vecs):
    """Computes an orthonormal basis for the given set of (complex) vectors.

    Vectors are expected to be supplied as the rows of an array, i.e. the first
    vector should be vecs[0] etc.  An array of the same form is returned.

    The orthonormalisation is done with respect to the inner product used in QM:
        <a|b> = a^dagger b,
    i.e. including a complex conjugation.

    The algorithm implemented is the modified Gram-Schmidt procedure, given in
    Numerical Methods for Large Eigenvalue Problems: Revised Edition, algorithm
    1.2."""
    result = np.zeros_like(vecs)
    r = np.linalg.norm(vecs[0])
    if r == 0.0:
        raise ArithmeticError("Vector with norm 0 occured.")
    else:
        result[0] = vecs[0] / r
    for j in range(1, vecs.shape[0]):
        for i in range(j):
            rij = np.conj(result[i]) @ vecs[j]
            vecs[j] = vecs[j] - rij * result[i]
        rjj = np.linalg.norm(vecs[j])
        if rjj == 0.0:
            raise ArithmeticError("Vector with norm 0 occured.")
        else:
            result[j] = vecs[j] / rjj
    return result
