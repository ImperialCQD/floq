import collections

Eigensystem = collections.namedtuple('Eigensystem', (
                                         'frequency',
                                         'quasienergies',
                                         'k_eigenvectors',
                                         'initial_floquet_bras',
                                         'abstract_ket_coefficients',
                                         'k_derivatives',
                                    ))

# I use this custom sparse column representation of a matrix for compatibility
# with `numba`, since it can't understand `scipy.sparse` matrices.  I couple
# this with a simple implementation of a left dot product (i.e. vector . matrix)
# to get a speed up on the heaviest part of `du_dcontrols()`.
ColumnSparseMatrix = collections.namedtuple('ColumnSparseMatrix',
                                            ('in_column', 'row', 'value'))
