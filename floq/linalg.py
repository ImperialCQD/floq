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

def transfer_fidelity(u, initial, final):
    """Compute how well the unitary u transfers an initial state |i> to a final
    state |f>, quantified by fid = |<f| u |i>|^2 = <f| u |i><i| u |f>.

    Note that initial and final should be supplied as kets."""
    return np.abs(inner(final, u, initial))**2

def d_transfer_fidelity(u, dus, initial, final):
    """Calculate the gradient of the transfer fidelity:
        fid' = (<f|u|i><i|u|f>)'z
             = <f|u'|i><i|u|f> + <f|u|i><i|u'|f>
             = 2 Re(<f|u'|i><i|u|f>)."""
    iuf = inner(initial, np.conj(u.T), final)
    fui = inner(final, u, initial)
    return np.real(np.array([inner(final, du, initial) * iuf +
                             inner(initial, np.conj(du.T), final) * fui
                             for du in dus]))

def transfer_distance(u, initial, final):
    """Version of the transfer fidelity that is minimal when the
    transfer is ideal."""
    return 1 - transfer_fidelity(u, initial, final)

def d_transfer_distance(u, dus, initial, final):
    """Gradient of the transfer distance."""
    return -d_transfer_fidelity(u, dus, initial, final)

def operator_fidelity(u, target):
    """Calculate the operator fidelity between the unitaries
    u and target, defined as 1/dim * Re(trace(target^\dagger u)).

    This quantity is maximal when u = target, when it is equal to 1."""
    return hilbert_schmidt_product(target, u).real / u.shape[0]

def d_operator_fidelity(u, dus, target):
    """Calculate the gradient of the operator fidelity."""
    return np.array([operator_fidelity(du, target) for du in dus])

def operator_distance(u, target):
    """Calculate a quantity proportional to the Hilbert-Schmidt distance
        tr((u - target)(u^\dagger - target^\dagger))
            = 2 (dim - Re(tr(target^\dagger u))),
    to be precise the following quantity:
        1.0 - Re(trace(target^\dagger u)) / dim.

    This quantity is minimised when u = target, which makes it useful for use
    with the minimisation routines built into SciPy."""
    return 1.0 - operator_fidelity(u, target)

def d_operator_distance(u, dus, target):
    """Calculate the gradient of the operator distance."""
    return -d_operator_fidelity(u, dus, target)

def inner(left, operator, right):
    """Compute <left|operator|right>."""
    leftconj = np.transpose(np.conj(left))
    return np.conj(left.T) @ operator @ right

def hilbert_schmidt_product(a, b):
    """Compute the Hilbert-Schmidt inner product between operators a and b."""
    return np.trace(np.conj(a.T) @ b)
