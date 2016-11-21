import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import floq.helpers as h
import floq.blockmatrix as bm
import floq.fixed_system as fs
import floq.errors as errors
import itertools
import copy
import cmath
from numba import autojit


def do_evolution(hf, params):
    """
    Calculate the time evolution operator U
    given a Fourier transformed Hamiltonian Hf
    """
    k = build_k(hf, params)

    vals, vecs = find_eigensystem(k, params)

    phi = calculate_phi(vecs)
    psi = calculate_psi(vecs, params)

    return calculate_u(phi, psi, vals, params)


def do_evolution_with_derivatives(hf, dhf, params):
    """
    Calculate the time evolution operator U
    given a Fourier transformed Hamiltonian Hf,
    as well as its derivative dU given dHf
    """
    k = build_k(hf, params)

    vals, vecs = find_eigensystem(k, params)

    phi = calculate_phi(vecs)
    psi = calculate_psi(vecs, params)

    u = calculate_u(phi, psi, vals, params)

    dk = build_dk(dhf, params)

    du = calculate_du(dk, psi, vals, vecs, params)

    return [u, du]



def build_k(hf, p):
    return numba_build_k(hf, p.dim, p.k_dim, p.nz, p.nc, p.omega)


@autojit(nopython=True)
def numba_build_k(hf, dim, k_dim, nz, nc, omega):
    hf_max = (nc-1)/2
    k = h.numba_zeros((k_dim, k_dim))

    # Assemble K by placing each component of Hf in turn, which
    # for a fixed Fourier index lie on diagonals, with 0 on the
    # main diagonal, positive numbers on the right and negative on the left
    #
    # The first row is therefore essentially Hf(0) Hf(1) ... Hf(hf_max) 0 0 0 ...
    # The last row is then ... 0 0 0 Hf(-hf_max) ... Hf(0)
    # Note that the main diagonal acquires a factor of omega*identity*(row/column number)

    for n in range(-hf_max, hf_max+1):
        start_row = max(0, n)  # if n < 0, start at row 0
        start_col = max(0, -n)  # if n > 0, start at col 0

        stop_row = min((nz-1)+n, nz-1)
        stop_col = min((nz-1)-n, nz-1)

        row = start_row
        col = start_col

        current_component = hf[h.n_to_i(n, nc)]

        while row <= stop_row and col <= stop_col:
            if n == 0:
                block = current_component + np.identity(dim)*omega*h.i_to_n(row, nz)
                bm.set_block_in_matrix(block, k, dim, nz, row, col)
            else:
                bm.set_block_in_matrix(current_component, k, dim, nz, row, col)

            row += 1
            col += 1

    return k


def build_dk(dhf, p):
    return numba_build_dk(dhf, p.np, p.dim, p.k_dim, p.nz, p.nc)


@autojit(nopython=True)
def numba_build_dk(dhf, npm, dim, k_dim, nz, nc):
    dk = np.empty((npm, k_dim, k_dim), dtype=np.complex128)
    for c in range(npm):
        dk[c, :, :] = numba_build_k(dhf[c], dim, k_dim, nz, nc, 0.0)

    return dk


def find_eigensystem(k, p):
    # Find eigenvalues and eigenvectors for k,
    # identify the dim unique ones,
    # return them in a segmented form
    vals, vecs = compute_eigensystem(k, p)

    unique_vals = find_unique_vals(vals, p)

    vals = vals.round(p.decimals)
    indices_unique_vals = [np.where(vals == eva)[0][0] for eva in unique_vals]

    unique_vecs = np.array([vecs[:, i] for i in indices_unique_vals])

    unique_vecs = separate_components(unique_vecs, p.nz)

    return [unique_vals, unique_vecs]


def compute_eigensystem(k, p):
    # Find eigenvalues and eigenvectors of k,
    # using the method specified in the parameters
    if p.sparse:
        k = sp.csc_matrix(k)

        number_of_eigs = min(2*p.dim, p.k_dim)
        vals, vecs = la.eigs(k, k=number_of_eigs, sigma=0.0)
    else:
        vals, vecs = np.linalg.eig(k)
        vals, vecs = trim_eigensystem(vals, vecs, p)

    vals = vals.real.astype(np.float64, copy=False)

    return vals, vecs


def trim_eigensystem(vals, vecs, p):
    # Trim eigenvalues and eigenvectors to only 2*dim ones
    # clustered around zero

    # Sort eigenvalues and -vectors in increasing order
    idx = vals.argsort()
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Only keep values around 0
    middle = p.k_dim/2
    cutoff_left = max(0, middle - p.dim)
    cutoff_right = min(p.k_dim, cutoff_left + 2*p.dim)

    cut_vals = vals[cutoff_left:cutoff_right]
    cut_vecs = vecs[:, cutoff_left:cutoff_right]

    return cut_vals, cut_vecs


def find_unique_vals(vals, p):
    # In the list of values supplied, find the set of dim
    # e_i that fulfil (e_i - e_j) mod omega != 0 for all i,j,
    # and that lie closest to 0.

    mod_vals = np.mod(vals, p.omega)
    mod_vals = mod_vals.round(decimals=p.decimals)  # round to suppress floating point issues

    unique_vals = np.unique(mod_vals)

    # the unique_vals are ordered and >= 0, but we'd rather have them clustered around 0
    should_be_negative = np.where(unique_vals > p.omega/2.)
    unique_vals[should_be_negative] = (unique_vals[should_be_negative]-p.omega).round(p.decimals)

    if unique_vals.shape[0] != p.dim:
        raise errors.EigenvalueNumberError(vals, unique_vals)
    else:
        return np.sort(unique_vals)


def separate_components(vecs, n):
    # Given an array of vectors vecs,
    # return an array of each of the vectors split into n sub-arrays

    return np.array([np.split(eva, n) for eva in vecs])



@autojit(nopython=True)
def calculate_phi(vecs):
    # Given an array of eigenvectors vecs,
    # sum over all frequency components in each
    dim = vecs.shape[0]
    phi = np.empty((dim, dim), dtype=np.complex128)
    for i in range(dim):
        phi[i] = numba_sum_components(vecs[i], dim)
    return phi


@autojit(nopython=True)
def numba_sum_components(vec, dim):
    n = vec.shape[0]
    result = h.numba_zeros(dim)
    for i in range(n):
        result += vec[i]
    return result



def calculate_psi(vecs, p):
    # Given an array of eigenvectors vecs,
    # sum over all frequency components in each,
    # weighted by exp(- i omega t n), with n
    # being the Fourier index of the component

    return numba_calculate_psi(vecs, p.dim, p.nz, p.omega, p.t)


@autojit(nopython=True)
def numba_calculate_psi(vecs, dim, nz, omega, t):
    # Given an array of eigenvectors vecs,
    # sum over all frequency components in each,
    # weighted by exp(- i omega t n), with n
    # being the Fourier index of the component

    psi = h.numba_zeros((dim, dim))

    for k in range(0, dim):
        partial = h.numba_zeros(dim)
        for i in range(0, nz):
            num = h.i_to_n(i, nz)
            partial += np.exp(1j*omega*t*num)*vecs[k][i]
        psi[k, :] = partial

    return psi



def calculate_u(phi, psi, energies, p):
    u = np.zeros([p.dim, p.dim], dtype='complex128')
    t = p.t

    for k in xrange(0, p.dim):
        u += np.exp(-1j*t*energies[k])*np.outer(psi[k], np.conj(phi[k]))

    return u


def calculate_du(dk, psi, vals, vecs, p):
    # Inside the loops, we avoid accessing complicated objects
    dim = p.dim
    nz_max = p.nz_max
    nz = p.nz
    npm = p.np
    omega = p.omega
    t = p.t

    vecsstar = np.conj(vecs)

    factors = calculate_factors(dk, nz, nz_max, dim, npm, vals, vecs, vecsstar, omega, t)

    return assemble_du(nz, nz_max, dim, npm, factors, psi, vecsstar)


def calculate_factors(dk, nz, nz_max, dim, npm, vals, vecs, vecsstar, omega, t):
    factors = np.empty([npm, 2*nz+1, dim, dim], dtype=np.complex128)

    for dn in xrange(-nz_max*2, 2*nz_max+1):
        idn = h.n_to_i(dn, 2*nz)
        for i1 in xrange(0, dim):
            for i2 in xrange(0, dim):
                v1 = np.roll(vecsstar[i1], dn, axis=0)  # not supported by numba!
                for c in xrange(0, npm):
                    factors[c, idn, i1, i2] = (integral_factors(vals[i1], vals[i2], dn, omega, t) *
                                               expectation_value(dk[c], v1, vecs[i2]))

    return factors


@autojit(nopython=True)
def assemble_du(nz, nz_max, dim, npm, alphas, psi, vecsstar):
    du = h.numba_zeros((npm, dim, dim))

    for n2 in range(-nz_max, nz_max+1):
        for i1 in range(0, dim):
            for i2 in range(0, dim):
                product = h.numba_outer(psi[i1], vecsstar[i2, ((nz-1)/2-n2) % nz])
                for n1 in range(-nz_max, nz_max+1):
                    idn = h.n_to_i(n1-n2, 2*nz)
                    for c in xrange(0, npm):
                        du[c] += alphas[c, idn, i1, i2]*product

    return du


@autojit(nopython=True)
def integral_factors(e1, e2, dn, omega, t):
    if e1 == e2 and dn == 0:
        return -1.0j*cmath.exp(-1j*t*e1)*t
    else:
        return (cmath.exp(-1j*t*e1)-cmath.exp(-1j*t*(e2-omega*dn)))/((e1-e2+omega*dn))


@autojit(nopython=True)
def expectation_value(dk, v1, v2):
    a = v1.flatten()
    b = v2.flatten()

    return np.dot(np.dot(a, dk), b)