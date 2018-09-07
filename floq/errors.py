class EigenvalueNumberError(Exception):
    def __init__(self, all_vals, unique_vals):
        self.all_vals, self.unique_vals = all_vals, unique_vals

    def __str__(self):
        return "Number of eigenvalues of K does not match \
                dimension of the the Hilbert space. \n All vals: " \
                + repr(self.all_vals) + "\n 'Unique' vals: " \
                + repr(self.unique_vals)

class NZTooLargeError(Exception):
    """Thrown when NZ gets too large"""
    pass
