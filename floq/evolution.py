import numpy as np
import floq.helpers as h
import floq.blockmatrix as bm
from IPython import embed

class FloquetError(Exception):
    pass

def find_eigensystem(k,hf_dim,omega):
    """
    Find eigenvalues and eigenvectors for k,
    identify the hf_dim unique ones
    """
    evas, eves = np.linalg.eig(k)
    unique_evas = find_unique_evas(evas,hf_dim,omega)
    indices_unique_evas = [np.where(evas==eva)[0][0] for eva in unique_evas]
    unique_eves = [eves[i] for i in indices_unique_evas]

    return [unique_evas,unique_eves]

def find_unique_evas(evas,hf_dim,omega):
    """
    In the list of values supplied, find the set of hf_dim 
    e_i that fulfil (e_i - e_j) mod omega != 0 for all i,j,
    and that lie closest to 0.
    """
    mod_evas = np.mod(evas,omega)
    mod_evas = mod_evas.round(decimals=10)

    unique_evas = np.unique(mod_evas) 

    # the unique_evas are ordered and >= 0, but we'd rather have them clustered around 0
    should_be_negative = np.where(unique_evas>omega/2.)
    unique_evas[should_be_negative] = (unique_evas[should_be_negative]-omega).round(10)

    if unique_evas.shape[0] != hf_dim:
        raise FloquetError("Number of unique eigenvalues of K is not hf_dim. Spectrum possibly degenerate?")
    else:
        return np.sort(unique_evas)


def build_k(hf,n_zones,omega):
    """
    Build the Floquet-Hamiltonian K 
    from the Fourier transform of the system Hamiltonian
    """
    n_comp = hf.shape[0]
    hf_cutoff = (n_comp-1)/2

    hf_dim = hf.shape[1]
    
    k_dim = hf_dim*n_zones
    k_cutoff = (n_zones-1)/2

    k = np.zeros([k_dim,k_dim])

    # Assemble K by placing each component of Hf in turn
    # The components lie on diagonals, with Hf(0) on the main diagonal
    # The first row is therefore essentially Hf(0) Hf(1) ... Hf(hf_cutoff) 0 0 0 ...
    # The last row is then ... 0 0 0 Hf(-hf_cutoff) ... Hf(0)
    for num in xrange(-hf_cutoff,hf_cutoff+1):
        start_row = max(0,num) # num < 0, start at row 0
        start_col = max(0,-num) # num > 0, start at col 0
        
        stop_row = min((n_zones-1)+num,n_zones-1) # if num > 0, start from the last col
        stop_col = min((n_zones-1)-num,n_zones-1) # if num < 0, start from the last row

        row = start_row
        col = start_col

        hf_of_num = hf[h.num_to_i(num,n_comp)]

        while row <= stop_row and col <= stop_col:
            if num == 0:
                block = hf_of_num + np.identity(hf_dim)*omega*h.i_to_num(row,n_zones)
                bm.set_block_in_matrix(block,k,hf_dim,n_zones,row,col)
            else:
                bm.set_block_in_matrix(hf_of_num,k,hf_dim,n_zones,row,col)

            row += 1
            col += 1

    return k