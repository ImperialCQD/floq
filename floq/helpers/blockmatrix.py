import numba

@numba.jit(nopython=True)
def get_block_from_matrix(matrix, dim_block, n_block, row, col):
    start_row = row * dim_block
    start_col = col * dim_block
    stop_row = start_row + dim_block
    stop_col = start_col + dim_block
    return matrix[start_row:stop_row, start_col:stop_col]

@numba.jit(nopython=True)
def set_block_in_matrix(block, matrix, dim_block, n_block, row, col):
    start_row = row * dim_block
    start_col = col * dim_block
    stop_row = start_row + dim_block
    stop_col = start_col + dim_block
    matrix[start_row:stop_row, start_col:stop_col] = block
