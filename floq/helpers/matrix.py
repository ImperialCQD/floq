import numpy as np

def is_unitary(u, tolerance=1e-10):
    """Return True if u^dagger u is equal to the unit matrix with the given
    tolerance."""
    digits = int(-np.log10(tolerance))
    dim = u.shape[0]
    unitary = np.eye(dim, dtype=np.complex128)
    umat = np.mat(u)
    product = umat.H * umat
    product = np.round(product, digits - 1)  # required for some edge cases
    return np.allclose(product, unitary, atol=tolerance)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """Test if a and b are close with given relative and absolute precision."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def adjoint(m):
    """Compute the Hermitian adjoint."""
    return np.transpose(np.conj(m))

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
    r = norm(vecs[0])
    if r == 0.0:
        raise ArithmeticError("Vector with norm 0 occured.")
    else:
        result[0] = vecs[0] / r
    for j in range(1, vecs.shape[0]):
        for i in range(j):
            rij = product(result[i], vecs[j])
            vecs[j] = vecs[j] - rij * result[i]
        rjj = norm(vecs[j])
        if rjj == 0.0:
            raise ArithmeticError("Vector with norm 0 occured.")
        else:
            result[j] = vecs[j] / rjj
    return result

def product(a, b):
    """Compute <a | b>.  Includes complex conjugation!"""
    return np.conj(a).dot(b)

def norm(a):
    """Compute sqrt(<a|a>)."""
    return np.sqrt(product(a, a))
