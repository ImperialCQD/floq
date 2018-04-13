import numpy as np
from ..helpers.matrix import adjoint

def transfer_fidelity(u, initial, final):
    """Compute how well the unitary u transfers an initial state |i> to a final
    state |f>, quantified by fid = |<f| u |i>|^2 = <f| u |i><i| u |f>.

    Note that initial and final should be supplied as kets."""
    return np.abs(expectation_value(final, u, initial))**2

def d_transfer_fidelity(u, dus, initial, final):
    """Calculate the gradient of the transfer fidelity:
        fid' = (<f|u|i><i|u|f>)'z
             = <f|u'|i><i|u|f> + <f|u|i><i|u'|f>
             = 2 Re(<f|u'|i><i|u|f>)."""
    iuf = expectation_value(initial, adjoint(u), final)
    fui = expectation_value(final, u, initial)
    return np.real(np.array([expectation_value(final, du, initial) * iuf +
                             expectation_value(initial, adjoint(du), final) * fui
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

def expectation_value(left, operator, right):
    """Compute <left|operator|right>."""
    leftconj = np.transpose(np.conj(left))
    return np.dot(np.conj(left).T, np.dot(operator, right))

def hilbert_schmidt_product(a, b):
    """Compute the Hilbert-Schmidt inner product between operators a and b."""
    return np.trace(np.dot(np.matrix(a).getH(), b))
