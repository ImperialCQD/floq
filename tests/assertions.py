from unittest import TestCase
import numpy as np

class CustomAssertions(TestCase):
    def assertArrayEqual(self, a, b, decimals=5):
        if not np.allclose(a, b, atol=0.1**decimals):
            print(a.round(decimals))
            print(b.round(decimals))
            raise AssertionError('Arrays are not equal!')

    def assertAlmostEqualWithDecimals(self, a, b, decimals=5):
        if not np.isclose(a, b, atol=0.1**decimals):
            print(np.round(a, decimals))
            print(np.round(b, decimals))
            raise AssertionError('a and b are not equal!')
