from numba import autojit

@autojit
def n_to_i(num, n):
    """Translate num, ranging from -(n-1)/2 through (n-1)/2 into an index i from
    0 to n-1.  If num > (n-1)/2, map it into the interval.  This is necessary to
    translate from a physical Fourier mode number to an index in an array."""
    return (num + (n - 1) // 2) % n

@autojit
def i_to_n(i, n):
    """Translate index i, ranging from 0 to n-1 into a number from -(n-1)/2
    through (n-1)/2.  This is necessary to translate from an index to a physical
    Fourier mode number."""
    return i - (n - 1) // 2

@autojit
def make_even(n):
    return n + (n & 1)
