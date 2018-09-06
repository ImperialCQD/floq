import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
from ..helpers.index import n_to_i, i_to_n
from ..helpers.numpy_replacements import numba_outer, numba_zeros
from ..helpers import blockmatrix as bm
from ..helpers import matrix as mm
from numba import autojit

def get_u(hf, params):
    """Calculate the time evolution operator U, given a Fourier transformed
    Hamiltonian Hf and the parameters of the problem."""
    return get_u_and_eigensystem(hf, params)[0]

def get_u_and_udot(hf, params):
    """
    Calculate the time evolution operator U, given a Fourier transformed
    Hamiltonian Hf and the parameters of the problem, as well as its time
    derivative.
    """
    u, vals, vecs, phi, psi = get_u_and_eigensystem(hf, params)
    psidot = calculate_psidot(vecs, params)
    udot = calculate_udot(phi, psi, psidot, vals, params)
    return u, udot

def get_u_and_du(hf, dhf, params):
    """Calculate the time evolution operator U given a Fourier transformed
    Hamiltonian Hf, as well as its derivative dU given dHf, and the parameters
    of the problem."""
    u, vals, vecs, phi, psi = get_u_and_eigensystem(hf, params)
    du = get_du_from_eigensystem(dhf, psi, vals, vecs, params)
    return u, du

def get_u_and_eigensystem(hf, params):
    """Calculate the time evolution operator U, given a Fourier transformed
    Hamiltonian Hf and the parameters of the problem, and return it as well as
    the intermediary results."""
    k = assemble_k(hf, params)
    vals, vecs = find_eigensystem(k, params)
    phi = calculate_phi(vecs)
    psi = calculate_psi(vecs, params)
    return calculate_u(phi, psi, vals, params), vals, vecs, phi, psi

def get_du_from_eigensystem(dhf, psi, vals, vecs, params):
    dk = assemble_dk(dhf, params)
    return calculate_du(dk, psi, vals, vecs, params)

def get_udot_from_eigensystem(phi, psi, vals, vecs, params):
    """
    Calculate the time evolution operator U, given a Fourier transformed
    Hamiltonian Hf and the parameters of the problem, as well as its time
    derivative.
    """
    psidot = calculate_psidot(vecs, params)
    return calculate_udot(phi, psi, psidot, vals, params)

def assemble_k(hf, p):
    """Assemble the Floquet Hamiltonian K from the components of the
    Fourier-transformed Hamiltonian."""
    return numba_assemble_k(hf, p.dim, p.k_dim, p.nz, p.nc, p.omega)

@autojit(nopython=True)
def numba_assemble_k(hf, dim, k_dim, nz, nc, omega):
    """Assemble K by placing each component of Hf in turn, which for a fixed
    Fourier index lie on diagonals, with 0 on the
    main diagonal, positive numbers on the right and negative on the left

    The first row is therefore essentially
        Hf(0) Hf(-1) ... Hf(-hf_max) 0 0 0 ...
    The last row is then
        ... 0 0 0 Hf(+hf_max) ... Hf(0)
    Note that the main diagonal acquires a factor of
        omega * identity * row / column."""
    hf_max = (nc - 1) // 2
    k = numba_zeros((k_dim, k_dim))
    for n in range(-hf_max, hf_max + 1):
        current = hf[n_to_i(n, nc)]
        row = max(0, n)  # if n < 0, start at row 0
        col = max(0, -n)  # if n > 0, start at col 0
        stop_row = min(nz - 1 + n, nz - 1)
        stop_col = min(nz - 1 - n, nz - 1)
        while row <= stop_row and col <= stop_col:
            if n == 0:
                block = current + np.eye(dim) * omega * i_to_n(row, nz)
                bm.set_block_in_matrix(block, k, dim, nz, row, col)
            else:
                bm.set_block_in_matrix(current, k, dim, nz, row, col)
            row = row + 1
            col = col + 1
    return k

def assemble_dk(dhf, p):
    """Assemble the derivative of the Floquet Hamiltonian K from the components
    of the derivative of the Fourier-transformed Hamiltonian This is equivalent
    to K, with Hf -> d HF and omega -> 0."""
    return numba_assemble_dk(dhf, p.np, p.dim, p.k_dim, p.nz, p.nc)

@autojit(nopython=True)
def numba_assemble_dk(dhf, npm, dim, k_dim, nz, nc):
    dk = np.empty((npm, k_dim, k_dim), dtype=np.complex128)
    for c in range(npm):
        dk[c, :, :] = numba_assemble_k(dhf[c], dim, k_dim, nz, nc, 0.0)
    return dk

def find_eigensystem(k, p):
    # Find unique eigenvalues and -vectors,
    # return them as segments (each of which is a ket)
    unique_vals, unique_vecs = get_basis(k, p)
    unique_vecs = np.array([np.split(unique_vecs[i], p.nz)\
                            for i in range(p.dim)])
    return unique_vals, unique_vecs

def get_basis(k, p):
    """Compute the eigensystem of K, then separate out the dim relevant parts,
    orthogonalising degenerate subspaces."""
    vals, vecs = compute_eigensystem(k, p)
    start = find_first_above_value(vals, -0.5 * p.omega)
    picked_vals = vals[start:start + p.dim]
    picked_vecs = np.array([vecs[:, i] for i in range(start, start + p.dim)])
    for duplicate_set in find_duplicates(picked_vals, p.decimals):
        picked_vecs[duplicate_set] = mm.gram_schmidt(picked_vecs[duplicate_set])
    return picked_vals, picked_vecs

def compute_eigensystem(k, p):
    """Find eigenvalues and eigenvectors of k, using the method specified in the
    parameters (sparse is almost always faster, and is the default)."""
    if p.sparse:
        k = sp.csc_matrix(k)
        number_of_eigs = min(2 * p.dim, p.k_dim)
        # find number_of_eigs eigenvectors/-values around 0.0
        # -> trimming/sorting the eigensystem is NOT necessary
        vals, vecs = la.eigs(k, k=number_of_eigs, sigma=0.0)
    else:
        vals, vecs = trim_eigensystem(*np.linalg.eig(k), p)
    vals = vals.real.astype(np.float64, copy=False)
    # sort eigenvalues / eigenvectors
    idx = vals.argsort()
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs

def trim_eigensystem(vals, vecs, p):
    """Trim eigenvalues and eigenvectors to only 2*dim ones clustered around
    zero."""
    # Sort eigenvalues and -vectors in increasing order.
    idx = vals.argsort()
    vals = vals[idx]
    vecs = vecs[:, idx]
    # Only keep values around 0
    middle = p.k_dim // 2
    cutoff_left = max(0, middle - p.dim)
    cutoff_right = min(p.k_dim, cutoff_left + 2 * p.dim)
    cut_vals = vals[cutoff_left:cutoff_right]
    cut_vecs = vecs[:, cutoff_left:cutoff_right]
    return cut_vals, cut_vecs

@autojit(nopython=True)
def find_first_above_value(array, value):
    """Find the index of the first array entry > value."""
    for i, array_value in enumerate(array):
        if array_value > value:
            return i
    return None

def find_duplicates(array, decimals):
    """
    Given a sorted 1D array of values, return an iterator where each element
    corresponds to one degenerate eigenvalue, and the element is a list of
    indices where that eigenvalue occurs.

    For example,
        find_duplicates([0, 0, 0, 1, 2, 2, 3])
    returns
        [[0, 1, 2], [4, 5]].

    Arguments --
    array: 1D sorted np.array -- The array to find duplicates in.
    decimals: float --
        The number of decimal places to round `array` to when comparing values
        for equality.

    Returns --
    duplicate_sets: iterator of np.array of int --
        An iterator yielding arrays of the indices of each duplicate entry.
        Entries which are not duplicated will not be referenced in the output.
    """
    indices = np.arange(array.shape[0])
    _, start_indices = np.unique(np.round(array, decimals=decimals),
                                 return_index=True)
    # start_indices will always contain 0 first, but np.split doesn't need it.
    return filter(lambda x: x.size > 1, np.split(indices, start_indices[1:]))

@autojit(nopython=True)
def calculate_phi(vecs):
    """Given an array of eigenvectors vecs, sum over Fourier components in
    each."""
    dim = vecs.shape[0]
    phi = np.empty((dim, dim), dtype=np.complex128)
    for i in range(dim):
        phi[i] = numba_sum_components(vecs[i], dim)
    return phi

@autojit(nopython=True)
def numba_sum_components(vec, dim):
    n = vec.shape[0]
    result = numba_zeros(dim)
    for i in range(n):
        result += vec[i]
    return result

def calculate_psi(vecs, p):
    """Given an array of eigenvectors vecs, sum over all Fourier components in
    each, weighted by exp(- i omega t n), with n being the Fourier index of the
    component."""
    return numba_calculate_psi(vecs, p.dim, p.nz, p.omega, p.t)

@autojit(nopython=True)
def numba_calculate_psi(vecs, dim, nz, omega, t):
    psi = numba_zeros((dim, dim))
    for k in range(dim):
        partial = numba_zeros(dim)
        for i in range(nz):
            num = i_to_n(i, nz)
            partial += np.exp(1j * omega * t * num) * vecs[k][i]
        psi[k, :] = partial
    return psi

def calculate_psidot(vecs, p):
    """
    Given an array of eigenvectors vecs, sum over all Fourier components in
    each, weighted by exp(- i omega t n), with n being the Fourier index of the
    component.
    """
    return numba_calculate_psidot(vecs, p.dim, p.nz, p.omega, p.t)

@autojit(nopython=True)
def numba_calculate_psidot(vecs, dim, nz, omega, t):
    psidot = numba_zeros((dim, dim))
    for k in range(0, dim):
        partial = numba_zeros(dim)
        for i in range(nz):
            num = i_to_n(i, nz)
            partial += (1j*omega*num) * np.exp(1j*omega*t*num) * vecs[k][i]
        psidot[k, :] = partial
    return psidot

def calculate_u(phi, psi, energies, p):
    u = np.zeros((p.dim, p.dim), dtype=np.complex128)
    t = p.t
    for k in range(p.dim):
        u += np.exp(-1j * t * energies[k]) * np.outer(psi[k], np.conj(phi[k]))
    return u

def calculate_udot(phi, psi, psidot, energies, p):
    udot = np.zeros((p.dim, p.dim), dtype=np.complex128)
    t = p.t
    for k in range(p.dim):
        udot += np.exp(-1j*t*energies[k]) * np.outer(psidot[k], np.conj(phi[k]))
        udot += -1j * energies[k] * np.exp(-1j*t*energies[k])\
                * np.outer(psi[k], np.conj(phi[k]))
    return udot

def calculate_du(dk, psi, vals, vecs, p):
    """
    Given the eigensystem of K, and its derivative, perform the computations
    to get dU.

    This routine is optimised and quite hard to read, I recommend taking a look
    in the museum, which contains functionally equivalent, but much more
    readable versions.
    """
    dim = p.dim
    nz_max = p.nz_max
    nz = p.nz
    np_ = p.np
    omega = p.omega
    t = p.t
    vecsstar = np.conj(vecs)
    factors = calculate_factors(dk, nz, nz_max, dim, np_, vals, vecs,\
                                vecsstar, omega, t)
    return assemble_du(nz, nz_max, dim, np_, factors, psi, vecsstar)

def calculate_factors(dk, nz, nz_max, dim, np_, vals, vecs, vecsstar, omega, t):
    # Factors in the sum for dU that only depend on dn=n1-n2, and therefore
    # can be computed more efficiently outside the "full" loop
    factors = np.empty([np_, 2*nz+1, dim, dim], dtype=np.complex128)
    for dn in range(-2 * nz_max, 2 * nz_max + 1):
        idn = n_to_i(dn, 2 * nz)
        for i1 in range(dim):
            for i2 in range(dim):
                v1 = np.roll(vecsstar[i1], dn, axis=0) # not supported by numba!
                for c in range(np_):
                    factors[c, idn, i1, i2] =\
                        integral_factors(vals[i1], vals[i2], dn, omega, t)\
                        * expectation_value(dk[c], v1, vecs[i2])
    return factors

@autojit(nopython=True)
def assemble_du(nz, nz_max, dim, np_, alphas, psi, vecsstar):
    """Execute the sum defining dU, taking pre-computed factors into account."""
    du = numba_zeros((np_, dim, dim))
    for n2 in range(-nz_max, nz_max + 1):
        for i1 in range(dim):
            for i2 in range(dim):
                product = numba_outer(psi[i1], vecsstar[i2, n_to_i(-n2, nz)])
                for n1 in range(-nz_max, nz_max + 1):
                    idn = n_to_i(n1 - n2, 2 * nz)
                    for c in range(np_):
                        du[c] += alphas[c, idn, i1, i2] * product

    return du

@autojit(nopython=True)
def integral_factors(e1, e2, dn, omega, t):
    if e1 == e2 and dn == 0:
        return -1j * t * np.exp(-1j * t * e1)
    else:
        return (np.exp(-1j * t * e1) - np.exp(-1j *t * (e2 - omega * dn)))\
               / (dn * omega + e1 - e2)

@autojit(nopython=True)
def expectation_value(dk, v1, v2):
    """Computes <v1|dk|v2>, assuming v1 is already conjugated."""
    # v1 and v2 are split into Fourier components, we undo that here
    a = v1.flatten()
    b = v2.flatten()
    return np.dot(np.dot(a, dk), b)
