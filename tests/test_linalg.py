from unittest import TestCase
from tests.assertions import CustomAssertions
import numpy as np
import floq.linalg

class TestGetBlock(TestCase):
    def setUp(self):
        self.dim_block = 5
        self.n_block = 3

        self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i  \
            = [j*np.ones([self.dim_block, self.dim_block]) for j in range(9)]

        matrix = np.bmat([[self.a, self.b, self.c],
                         [self.d, self.e, self.f],
                         [self.g, self.h, self.i]])
        self.matrix = np.array(matrix)

    def test_a(self):
        block = floq.linalg.get_block(self.matrix, self.dim_block, self.n_block, 0, 0)
        self.assertTrue(np.array_equal(block, self.a))

    def test_b(self):
        block = floq.linalg.get_block(self.matrix, self.dim_block, self.n_block, 0, 1)
        self.assertTrue(np.array_equal(block, self.b))

    def test_c(self):
        block = floq.linalg.get_block(self.matrix, self.dim_block, self.n_block, 0, 2)
        self.assertTrue(np.array_equal(block, self.c))

    def test_d(self):
        block = floq.linalg.get_block(self.matrix, self.dim_block, self.n_block, 1, 0)
        self.assertTrue(np.array_equal(block, self.d))

    def test_e(self):
        block = floq.linalg.get_block(self.matrix, self.dim_block, self.n_block, 1, 1)
        self.assertTrue(np.array_equal(block, self.e))

    def test_f(self):
        block = floq.linalg.get_block(self.matrix, self.dim_block, self.n_block, 1, 2)
        self.assertTrue(np.array_equal(block, self.f))

    def test_g(self):
        block = floq.linalg.get_block(self.matrix, self.dim_block, self.n_block, 2, 0)
        self.assertTrue(np.array_equal(block, self.g))

    def test_h(self):
        block = floq.linalg.get_block(self.matrix, self.dim_block, self.n_block, 2, 1)
        self.assertTrue(np.array_equal(block, self.h))

    def test_i(self):
        block = floq.linalg.get_block(self.matrix, self.dim_block, self.n_block, 2, 2)
        self.assertTrue(np.array_equal(block, self.i))


class TestSetBlock(TestCase):
    def setUp(self):
        self.dim_block = 5
        self.n_block = 3

        self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i \
            = [j*np.ones([self.dim_block, self.dim_block]) for j in range(9)]

        matrix = np.bmat([[self.a, self.b, self.c],
                         [self.d, self.e, self.f],
                         [self.g, self.h, self.i]])
        self.original = np.array(matrix)

        total_size = self.dim_block*self.n_block
        self.copy = np.zeros([total_size,total_size])

    def test_set(self):
        # Try to recreate self.original with the new function
        floq.linalg.set_block(self.a, self.copy, self.dim_block, self.n_block, 0, 0)
        floq.linalg.set_block(self.b, self.copy, self.dim_block, self.n_block, 0, 1)
        floq.linalg.set_block(self.c, self.copy, self.dim_block, self.n_block, 0, 2)
        floq.linalg.set_block(self.d, self.copy, self.dim_block, self.n_block, 1, 0)
        floq.linalg.set_block(self.e, self.copy, self.dim_block, self.n_block, 1, 1)
        floq.linalg.set_block(self.f, self.copy, self.dim_block, self.n_block, 1, 2)
        floq.linalg.set_block(self.g, self.copy, self.dim_block, self.n_block, 2, 0)
        floq.linalg.set_block(self.h, self.copy, self.dim_block, self.n_block, 2, 1)
        floq.linalg.set_block(self.i, self.copy, self.dim_block, self.n_block, 2, 2)
        self.assertTrue(np.array_equal(self.copy,self.original))


class TestFourierIndexToNormalIndex(TestCase):
    def test_start(self):
        self.assertEqual(floq.linalg.n_to_i(-40, 81), 0)

    def test_end(self):
        self.assertEqual(floq.linalg.n_to_i(40, 81), 80)

    def test_middle(self):
        self.assertEqual(floq.linalg.n_to_i(0, 81), 40)

    def test_in_between(self):
        self.assertEqual(floq.linalg.n_to_i(-3, 81), 37)

    def test_too_big_a_bit(self):
        self.assertEqual(floq.linalg.n_to_i(5, 7), floq.linalg.n_to_i(-2, 7))

    def test_too_big_a_lot(self):
        self.assertEqual(floq.linalg.n_to_i(5+7, 7), floq.linalg.n_to_i(-2, 7))

    def test_too_small_a_bit(self):
        self.assertEqual(floq.linalg.n_to_i(-6, 7), floq.linalg.n_to_i(1, 7))

    def test_too_small_a_lot(self):
        self.assertEqual(floq.linalg.n_to_i(-6-14, 7), floq.linalg.n_to_i(1, 7))

class TestNormalIndexToFourierIndex(TestCase):
    def test_start(self):
        self.assertEqual(floq.linalg.i_to_n(0, 81), -40)

    def test_end(self):
        self.assertEqual(floq.linalg.i_to_n(80, 81), 40)

    def test_middle(self):
        self.assertEqual(floq.linalg.i_to_n(40, 81), 0)

    def test_in_between(self):
        self.assertEqual(floq.linalg.i_to_n(37, 81), -3)


class TestIsUnitary(TestCase):
    def test_true_if_unitary(self):
        u = np.array([[-0.288822 - 0.154483j, 0.20768 - 0.22441j, 0.0949032 - 0.0560178j, -0.385994 + 0.210021j, 0.423002 - 0.605778j, 0.135684 - 0.172261j], [0.0998628 - 0.364186j, 0.408817 - 0.35846j, -0.224508 - 0.550201j, 0.258427 + 0.263299j, -0.0297947 + 0.180679j, -0.0134853 + 0.197029j], [0.541087 - 0.216046j, -0.306777 + 0.0439077j, -0.479354 + 0.0395382j, -0.474755 + 0.264776j, -0.0971467 - 0.0167121j, 0.121192 - 0.115168j], [-0.0479833 - 0.133938j, 0.0696875 - 0.539678j, 0.314762 + 0.391157j, -0.376453 + 0.00569747j, -0.348676 + 0.2061j, 0.0588683 + 0.34972j], [-0.524482 + 0.213402j, 0.152127 + 0.111274j, -0.308402 - 0.134059j, -0.448647 + 0.120202j, -0.0680734 + 0.435883j, -0.295969 - 0.181141j], [-0.119405 + 0.235674j, 0.349453 + 0.247169j, -0.169971 + 0.0966179j, 0.0310919 + 0.129778j, -0.228356 + 0.00511762j, 0.793243 + 0.0977203j]])
        self.assertTrue(floq.linalg.is_unitary(u, 5))

    def test_true_if_not_unitary(self):
        u = np.array([[-5.288822 - 0.154483j, 0.20768 - 0.22441j, 0.0949032 - 0.0560178j, -0.385994 + 0.210021j, 0.423002 - 0.605778j, 0.135684 - 0.172261j], [0.0998628 - 0.364186j, 0.408817 - 0.35846j, -0.224508 - 0.550201j, 0.258427 + 0.263299j, -0.0297947 + 0.180679j, -0.0134853 + 0.197029j], [0.541087 - 0.216046j, -0.306777 + 0.0439077j, -0.479354 + 0.0395382j, -0.474755 + 0.264776j, -0.0971467 - 0.0167121j, 0.121192 - 0.115168j], [-0.0479833 - 0.133938j, 0.0696875 - 0.539678j, 0.314762 + 0.391157j, -0.376453 + 0.00569747j, -0.348676 + 0.2061j, 0.0588683 + 0.34972j], [-0.524482 + 0.213402j, 0.152127 + 0.111274j, -0.308402 - 0.134059j, -0.448647 + 0.120202j, -0.0680734 + 0.435883j, -0.295969 - 0.181141j], [-0.119405 + 0.235674j, 0.349453 + 0.247169j, -0.169971 + 0.0966179j, 0.0310919 + 0.129778j, -0.228356 + 0.00511762j, 0.793243 + 0.0977203j]])
        self.assertFalse(floq.linalg.is_unitary(u, 5))

class TestGramSchmidt(CustomAssertions):
    def setUp(self):
        self.array = np.array([[1.0j, 2.0, 3.0],
                               [0.0+0.2j, 1.0, 1.0],
                               [3.0, 2.0, 1.0]])
        self.res = floq.linalg.gram_schmidt(self.array)
        self.x = self.res[0]
        self.y = self.res[1]
        self.z = self.res[2]

    def test_orthogonality_x_y(self):
        self.assertAlmostEqual(np.conj(self.x) @ self.y, 0.0)

    def test_orthogonality_x_z(self):
        self.assertAlmostEqual(np.conj(self.x) @ self.z, 0.0)

    def test_orthogonality_y_z(self):
        print(self.y)
        print(self.z)
        self.assertAlmostEqual(np.conj(self.y) @ self.z, 0.0)

    def test_normalised_x(self):
        self.assertAlmostEqual(np.linalg.norm(self.x), 1.0)

    def test_normalised_y(self):
        self.assertAlmostEqual(np.linalg.norm(self.y), 1.0)

    def test_normalised_z(self):
        self.assertAlmostEqual(np.linalg.norm(self.z), 1.0)
