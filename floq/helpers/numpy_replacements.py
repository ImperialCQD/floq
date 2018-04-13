import numpy as np
from numba import autojit

@autojit(nopython=True)
def numba_zeros(dims):
    ary = np.empty(dims, dtype=np.complex128)
    ary[:] = 0j
    return ary

@autojit(nopython=True)
def numba_outer(a, b):
    m = a.shape[0]
    n = b.shape[0]
    result = np.empty((m, n), dtype=np.complex128)
    for i in range(m):
        for j in range(n):
            result[i, j] = a[i] * b[j]
    return result
