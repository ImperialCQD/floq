import numba
import numpy as np
import scipy.sparse.linalg
import collections
from .. import linalg

Eigensystem = collections.namedtuple('Eigensystem', (
                                         'frequency',
                                         'quasienergies',
                                         'k_eigenvectors',
                                         'initial_floquet_bras',
                                         'abstract_ket_coefficients',
                                         'k_derivatives',
                                    ))

def eigensystem(parameters, hamiltonian, dhamiltonian=None):
    """
    Calculate the time-invariant eigensystem of the Floquet system.  This needs
    to be recalculated whenever the Hamiltonian (or its derivatives) change, but
    not if the time changes.  The result of this can be passed to the other
    calculation routines.

    If the Hamiltonian's derivative is not supplied, then the derivatives of the
    `K` matrix won't be build, which saves time but prevents the usage of the
    `du_dcontrols()` function.

    Arguments --
    parameters: FixedSystemParameters --
        This will change in the future, because the new code mostly just infers
        all the parameters that are stored in it, and instantiating one of those
        classes requires a time, which the new code does not need.

    hamiltonian: 3D np.array of complex --
        This must be the matrix of a Hamiltonian, split into Fourier components
        in the same manner used in the return values of
        `floq.System.hamiltonian()`.

    dhamiltonian: 4D np.array of complex --
        Optionally, the matrix form of the derivatives of a Hamiltonian, as in
        the output of `floq.System.dhamiltonian()`.  If not supplied, then the
        resulting `Eigensystem` cannot be used with the `du_dcontrols()`
        function.

    Returns:
    Eigensystem --
        A collection of parameters that are not time-dependent, which can be
        passed to the time-specific functions.
    """
    n_zones = parameters.nz
    k = assemble_k(hamiltonian, parameters)
    quasienergies, k_eigenvectors = find_eigensystem(k, parameters)
    # Sum the eigenvectors along the Fourier-mode axis at `time = 0` to contract
    # the abstract Hilbert space back to the original one.
    initial_floquet_bras = np.conj(np.sum(k_eigenvectors, axis=1))
    fourier_modes = np.arange((1-n_zones)//2, 1 + (n_zones//2))
    abstract_ket_coefficients = 1j * parameters.omega * fourier_modes
    k_derivatives = None if dhamiltonian is None\
                    else assemble_dk(dhamiltonian, parameters)
    return Eigensystem(parameters.omega, quasienergies, k_eigenvectors,
                       initial_floquet_bras, abstract_ket_coefficients,
                       k_derivatives)

@numba.jit(nopython=True)
def current_floquet_kets(eigensystem, time):
    """
    Get the Floquet basis kets at a given time.  These are the
        |psi_j(t)> = exp(-i energy[j] t) |phi_j(t)>,
    using the notation in Marcel's thesis, equation (1.13).
    """
    weights = np.exp(time * eigensystem.abstract_ket_coefficients)
    weights = weights.reshape((1, -1, 1))
    return np.sum(weights * eigensystem.k_eigenvectors, axis=1)

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
def conjugate_rotate_into(out, input, amount):
    """
    Equivalent to `out = np.conj(np.roll(input, amount, axis=0))`, but `roll()`
    isn't supported by numba.  Also, we can directly write into `out` so we
    don't have any allocations in tight loops.
    """
    if amount < 0:
        out[:amount] = np.conj(input[abs(amount):])
        out[amount:] = np.conj(input[:abs(amount)])
    elif amount > 0:
        out[amount:] = np.conj(input[:-amount])
        out[:amount] = np.conj(input[-amount:])
    else:
        out[:] = np.conj(input)

@numba.jit(nopython=True)
def integral_factors(eigensystem, time):
    """
    Calculate the "integral factors" for use in the control-derivatives of the
    time-evolution operator.  These are the
        e(j, j'; delta mu)
    from equation (1.48) in Marcel's thesis.
    """
    # This function is comparatively long compared to the old version because it
    # does some questionable loop reorganisation to prevent `if` statements in
    # deeply nested loops.
    n_zones = eigensystem.k_eigenvectors.shape[1]
    dimension = eigensystem.k_eigenvectors.shape[2]
    energies = eigensystem.quasienergies
    frequency = eigensystem.frequency
    energy_phases = np.exp(-1j * time * energies)
    differences = np.arange(1.0 - n_zones, n_zones)
    diff_exponentials = np.exp(1j * time * frequency * differences)
    out = np.empty((differences.shape[0], dimension, dimension),
                   dtype=np.complex128)
    # Fill diagonal of 0 difference, then treat it specially to avoid an `if`
    # statement in a tightly nested loop.  Can be replaced by
    #   np.fill_diagonal(out[n_zones - 1], -1j * time * energy_phases)
    # once numba 0.40 is out.
    diag = -1j * time * energy_phases
    for i in range(energies.shape[0]):
        out[n_zones - 1, i, i] = diag[i]
    for i in range(energies.shape[0]):
        for j in range(i):
            value = (energy_phases[i] - energy_phases[j])\
                    / (energies[i] - energies[j])
            out[n_zones - 1, i, j] = out[n_zones - 1, j, i] = value
    # Handle all the rest of the values, making sure to avoid the zero
    # difference case.
    # Handle negative differences first.
    for diff_index in range(n_zones - 1):
        separation = frequency * differences[diff_index]
        exponential = diff_exponentials[diff_index]
        for i in range(energies.shape[0]):
            numer = energy_phases[i] - energy_phases*exponential
            denom = energies[i] + separation - energies
            out[diff_index, i] = numer / denom
    for diff_index in range(n_zones, differences.shape[0]):
        separation = frequency * differences[diff_index]
        exponential = diff_exponentials[diff_index]
        for i in range(energies.shape[0]):
            numer = energy_phases[i] - energy_phases*exponential
            denom = energies[i] + separation - energies
            out[diff_index, i] = numer / denom
    return out

@numba.jit(nopython=True)
def combined_factors(eigensystem, time):
    """
    Calculate the "combined factors" for use in the control-derivatives of the
    time evolution operator.  These are the
        f(j, j'; delta mu)
    from equations (1.50) and (2.7) in Marcel's thesis.
    """
    n_parameters = eigensystem.k_derivatives.shape[0]
    n_zones = eigensystem.k_eigenvectors.shape[1]
    dimension = eigensystem.k_eigenvectors.shape[2]
    factors = np.empty((2*n_zones - 1, dimension, dimension, n_parameters),
                       dtype=np.complex128)
    rolled_k_eigenbra = np.zeros((n_zones, dimension),
                                 dtype=np.complex128)
    k_eigenkets = eigensystem.k_eigenvectors
    energies = eigensystem.quasienergies
    integral_terms = integral_factors(eigensystem, time)
    for diff_index, diff in enumerate(range(1 - n_zones, n_zones)):
        for i in range(dimension):
            conjugate_rotate_into(rolled_k_eigenbra, k_eigenkets[i], diff)
            for j in range(dimension):
                for parameter in range(n_parameters):
                    expectation = rolled_k_eigenbra.ravel()\
                                  @ eigensystem.k_derivatives[parameter]\
                                  @ k_eigenkets[j].ravel()
                    factors[diff_index, i, j, parameter] =\
                        integral_terms[diff_index, i, j] * expectation
    return factors

@numba.jit(nopython=True)
def du_dcontrols(eigensystem, time):
    """
    Calculate the derivatives of time-evolution operator with respect to the
    control parameters of the Hamiltonian at a certain time, using a
    pre-computed eigensystem.  This is only possible if the eigensystem was
    created using the Hamiltonian derivatives as well.
    """
    n_parameters = eigensystem.k_derivatives.shape[0]
    n_zones, dimension = eigensystem.k_eigenvectors.shape[1:3]
    out = np.zeros((n_parameters, dimension, dimension), dtype=np.complex128)
    if n_parameters == 0:
        return out
    factors = combined_factors(eigensystem, time)
    current_kets = current_floquet_kets(eigensystem, time)
    k_eigenbras = np.conj(eigensystem.k_eigenvectors)
    # Keep an extra dimension so we can broadcast the multiplication of the
    # per-control-parameter factors with the outer product.  This probably just
    # gets optimised into another loop internally, but makes the code here a
    # little tidier (one fewer level of indentation!).
    projector = np.empty((1, dimension, dimension), dtype=np.complex128)
    for i in range(dimension):
        for j in range(dimension):
            for zone_i in range(n_zones - 1, -1, -1):
                np.outer(current_kets[i], k_eigenbras[j, zone_i], out=projector)
                for diff_i in range(zone_i, zone_i + n_zones):
                    factor = factors[diff_i, i, j].reshape(n_parameters, 1, 1)
                    out += factor * projector
    return out

def get_u(hf, params):
    """Calculate the time evolution operator U, given a Fourier transformed
    Hamiltonian Hf and the parameters of the problem."""
    return get_u_and_eigensystem(hf, params)[0]

def get_u_and_du_dt(hf, params):
    """
    Calculate the time evolution operator U, given a Fourier transformed
    Hamiltonian Hf and the parameters of the problem, as well as its time
    derivative.
    """
    u, vals, vecs, phi, psi = get_u_and_eigensystem(hf, params)
    psidot = calculate_psidot(vecs, params)
    udot = calculate_udot(phi, psi, psidot, vals, params)
    return u, udot

def get_u_and_du_dcontrols(hf, dhf, params):
    """Calculate the time evolution operator U given a Fourier transformed
    Hamiltonian Hf, as well as its derivative dU given dHf, and the parameters
    of the problem."""
    u, vals, vecs, phi, psi = get_u_and_eigensystem(hf, params)
    du = get_du_dcontrols_from_eigensystem(dhf, psi, vals, vecs, params)
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

def get_du_dcontrols_from_eigensystem(dhf, psi, vals, vecs, params):
    dk = assemble_dk(dhf, params)
    return calculate_du(dk, psi, vals, vecs, params)

def get_du_dt_from_eigensystem(phi, psi, vals, vecs, params):
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

@numba.jit(nopython=True)
def numba_assemble_k(hf, dim, k_dim, nz, nc, omega):
    hf_max = (nc - 1) // 2
    k = np.zeros((k_dim, k_dim), dtype=np.complex128)
    for n in range(-hf_max, hf_max + 1):
        current = hf[linalg.n_to_i(n, nc)]
        row = max(0, n)  # if n < 0, start at row 0
        col = max(0, -n)  # if n > 0, start at col 0
        stop_row = min(nz - 1 + n, nz - 1)
        stop_col = min(nz - 1 - n, nz - 1)
        while row <= stop_row and col <= stop_col:
            if n == 0:
                block = current + np.eye(dim) * omega * linalg.i_to_n(row, nz)
                linalg.set_block(block, k, dim, nz, row, col)
            else:
                linalg.set_block(current, k, dim, nz, row, col)
            row = row + 1
            col = col + 1
    return k

def assemble_dk(dhf, p):
    """Assemble the derivative of the Floquet Hamiltonian K from the components
    of the derivative of the Fourier-transformed Hamiltonian This is equivalent
    to K, with Hf -> d HF and omega -> 0."""
    return numba_assemble_dk(dhf, p.np, p.dim, p.k_dim, p.nz, p.nc)

@numba.jit(nopython=True)
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
    for duplicates in find_duplicates(picked_vals, p.decimals):
        picked_vecs[duplicates] = linalg.gram_schmidt(picked_vecs[duplicates])
    return picked_vals, picked_vecs

def compute_eigensystem(k, p):
    """Find eigenvalues and eigenvectors of k, using the method specified in the
    parameters (sparse is almost always faster, and is the default)."""
    if p.sparse:
        k = scipy.sparse.csc_matrix(k)
        number_of_eigs = min(2 * p.dim, p.k_dim)
        # find number_of_eigs eigenvectors/-values around 0.0
        # -> trimming/sorting the eigensystem is NOT necessary
        vals, vecs = scipy.sparse.linalg.eigs(k, k=number_of_eigs, sigma=0.0)
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

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
def calculate_phi(k_eigenvectors):
    """
    The `shape[1]` index runs over the Fourier modes.  This contracts the
    abstract extended Hilbert space back into the same dimension as the
    original (non-Fourier) Hamiltonian.
    """
    return np.sum(k_eigenvectors, axis=1)

def calculate_psi(vecs, p):
    """Given an array of eigenvectors vecs, sum over all Fourier components in
    each, weighted by exp(- i omega t n), with n being the Fourier index of the
    component."""
    return numba_calculate_psi(vecs, p.dim, p.nz, p.omega, p.t)

@numba.jit(nopython=True)
def numba_calculate_psi(vecs, dim, nz, omega, t):
    psi = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(dim):
        partial = np.zeros(dim, dtype=np.complex128)
        for i in range(nz):
            num = linalg.i_to_n(i, nz)
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

@numba.jit(nopython=True)
def numba_calculate_psidot(vecs, dim, nz, omega, t):
    psidot = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(0, dim):
        partial = np.zeros(dim, np.complex128)
        for i in range(nz):
            num = linalg.i_to_n(i, nz)
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
    factors = np.empty([np_, 2*nz - 1, dim, dim], dtype=np.complex128)
    for dn in range(-2 * nz_max, 2 * nz_max + 1):
        idn = linalg.n_to_i(dn, 2*nz - 1)
        for i1 in range(dim):
            for i2 in range(dim):
                v1 = np.roll(vecsstar[i1], dn, axis=0) # not supported by numba!
                for c in range(np_):
                    factors[c, idn, i1, i2] =\
                        integral_factor(vals[i1], vals[i2], dn, omega, t)\
                        * expectation_value(dk[c], v1, vecs[i2])
    return factors

@numba.jit(nopython=True)
def assemble_du(nz, nz_max, dim, np_, alphas, psi, vecsstar):
    """Execute the sum defining dU, taking pre-computed factors into account."""
    du = np.zeros((np_, dim, dim), dtype=np.complex128)
    for n2 in range(-nz_max, nz_max + 1):
        for i1 in range(dim):
            for i2 in range(dim):
                product = np.outer(psi[i1], vecsstar[i2, linalg.n_to_i(-n2,nz)])
                for n1 in range(-nz_max, nz_max + 1):
                    idn = linalg.n_to_i(n1 - n2, 2 * nz)
                    for c in range(np_):
                        du[c] += alphas[c, idn, i1, i2] * product

    return du

@numba.jit(nopython=True)
def integral_factor(e1, e2, dn, omega, t):
    if e1 == e2 and dn == 0:
        return -1j * t * np.exp(-1j * t * e1)
    else:
        return (np.exp(-1j * t * e1) - np.exp(-1j *t * (e2 - omega * dn)))\
               / (dn * omega + e1 - e2)

@numba.jit(nopython=True)
def expectation_value(dk, v1, v2):
    """Computes <v1|dk|v2>, assuming v1 is already conjugated."""
    # v1 and v2 are split into Fourier components, we undo that here
    a = v1.flatten()
    b = v2.flatten()
    return np.dot(np.dot(a, dk), b)
