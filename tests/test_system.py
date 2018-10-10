import unittest
from tests.assertions import CustomAssertions
import numpy as np
import floq

def transformed_matrix_equal(a, b):
    order = np.argsort(a.mode)
    if not np.array_equal(sorted(a.mode), sorted(b.mode)):
        return False
    for i in order:
        if not np.array_equal(a.matrix[i], b.matrix[i]):
            return False
    return True


class TestCanonicalOperatorForm(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[1.0, 0.5j], [-0.5j, 1.0]])
        self.b = np.array([[2.0, 0], [0, 1.5]], dtype=np.complex128)
        self.c = np.array([[1.0, -0.5j], [0.5j, 1.0]])
        mode_low = (-1, 0, 1)
        mode_high = (5, 6, 7)
        self.target_low = floq.types.TransformedMatrix(mode_low,
                                                       (self.a, self.b, self.c))
        self.target_high = floq.types.TransformedMatrix(mode_high,
                                                        (self.a, self.b, self.c))

    def test_3d_array(self):
        op = np.array([self.a, self.b, self.c])
        built = floq.system._canonicalise_operator(op)
        self.assertTrue(transformed_matrix_equal(built, self.target_low))

    def test_failure_3d_array(self):
        op = np.array([self.c, self.b, self.a])
        built = floq.system._canonicalise_operator(op)
        self.assertFalse(transformed_matrix_equal(built, self.target_low))

    def test_dict(self):
        op = {-1: self.a, 0: self.b, 1: self.c}
        built = floq.system._canonicalise_operator(op)
        self.assertTrue(transformed_matrix_equal(built, self.target_low))

    def test_failure_dict(self):
        op = {1: self.a, 0: self.b, -1: self.c}
        built = floq.system._canonicalise_operator(op)
        self.assertTrue(transformed_matrix_equal(built, self.target_low))

    def test_iter(self):
        op = zip((-1, 0, 1), (self.a, self.b, self.c))
        built = floq.system._canonicalise_operator(op)
        self.assertTrue(transformed_matrix_equal(built, self.target_low))

    def test_failure_iter(self):
        op = zip((-5, 0, 1), (self.a, self.b, self.c))
        built = floq.system._canonicalise_operator(op)
        self.assertFalse(transformed_matrix_equal(built, self.target_low))

    def test_nop(self):
        built = floq.system._canonicalise_operator(self.target_low)
        self.assertTrue(transformed_matrix_equal(built, self.target_low))

    def test_failure_nop(self):
        built = floq.system._canonicalise_operator(self.target_high)
        self.assertFalse(transformed_matrix_equal(built, self.target_low))

    def test_iter_high(self):
        op = zip((5, 6, 7), (self.a, self.b, self.c))
        built = floq.system._canonicalise_operator(op)
        self.assertTrue(transformed_matrix_equal(built, self.target_high))

    def test_error_non_int(self):
        op = zip((0.5, 1, 2), (self.a, self.b, self.c))
        with self.assertRaises(TypeError):
            floq.system._canonicalise_operator(op)

    def test_error_non_square(self):
        op = zip((-1, 1, 2), (self.a.reshape(4, 1), self.b, self.c))
        with self.assertRaises(ValueError):
            floq.system._canonicalise_operator(op)

    def test_error_not_2d(self):
        op = zip((1, 1, 2), (self.a.reshape(4), self.b, self.c))
        with self.assertRaises(ValueError):
            floq.system._canonicalise_operator(op)

    def test_error_array_not_3d(self):
        op = np.array([[self.a, self.b, self.c]])
        with self.assertRaises(ValueError):
            floq.system._canonicalise_operator(op)

    def test_error_invalid_input(self):
        with self.assertRaises(TypeError):
            floq.system._canonicalise_operator("hello, world")
