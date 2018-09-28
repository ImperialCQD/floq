import numba
import numpy as np

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
