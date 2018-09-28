from unittest import TestCase
from tests.assertions import CustomAssertions
import numpy as np
import floq


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


u1 = np.array([[-0.1756 + 0.499j, -0.673 + 0.3477j, 0.0341 + 0.381j],
               [0.4613 + 0.3206j, 0.176 - 0.5432j, 0.0996 + 0.5903j],
               [0.5754 - 0.2709j, -0.2611 + 0.1789j, -0.7014 + 0.0582j]])
u2 = np.array([[0.2079 + 0.2374j, 0.5508 + 0.0617j, 0.3313 + 0.6953j],
               [-0.0423 - 0.2927j, 0.8017 - 0.1094j, -0.3873 - 0.3284j],
               [0.5487 + 0.7155j, 0.0209 - 0.1939j, -0.2633 - 0.2822j]])
u3 = np.array([[0.7025 + 0.2537j, -0.3209 + 0.1607j, 0.2748 - 0.4877j],
               [0.1706 - 0.1608j, 0.2006 - 0.9041j, -0.0438 - 0.2924j],
               [0.0347 + 0.6212j, 0.1164 + 0.0017j, -0.7627 - 0.1326j]])
u4 = np.array([[-0.2251 - 0.6249j, -0.5262 - 0.4341j, -0.0811 - 0.2947j],
               [0.3899 - 0.567j, 0.1173 + 0.5961j, -0.3893 - 0.0762j],
               [-0.1855 - 0.2256j, 0.0326 + 0.4055j, 0.8249 - 0.2623j]])

v1 = np.array([0.348713 - 0.435703j, -0.245625 - 0.0546497j, 0.575875 + 0.54186j])
v2 = np.array([-0.541912 + 0.349275j, 0.432916 + 0.181694j, 0.304055 + 0.521019j])


class TestTransferFidelity(CustomAssertions):

    def test_transfer_fidelity(self):
        fid = floq.linalg.transfer_fidelity(u1, v2, v1)
        self.assertAlmostEqualWithDecimals(fid, 0.131584, 4)


class TestTransferDistance(CustomAssertions):

    def test_is_zero_if_identity(self):
        v = np.array([1.0, 1.0j])/1.41421
        fid = floq.linalg.transfer_distance(np.eye(2), v, v)
        self.assertAlmostEqualWithDecimals(fid, 0.0, 4)

    def test_is_zero_if_bit_flip(self):
        zero = np.array([1.0, 0.0])
        one = np.array([0.0, 1.0])
        flip = np.array([[0, 1], [1, 0]])
        fid = floq.linalg.transfer_distance(flip, zero, one)
        self.assertAlmostEqualWithDecimals(fid, 0.0, 4)



class TestOperatorFidelity(CustomAssertions):

    def test_operator_fidelity(self):
        fid = floq.linalg.operator_fidelity(u1, u2)
        self.assertAlmostEqualWithDecimals(fid, 0.0378906, 4)



class TestOperatorDistance(CustomAssertions):

    def test_zero_if_equal(self):
        self.assertAlmostEqualWithDecimals(0.0, floq.linalg.operator_distance(u1, u1))

    def test_pos_if_nonequal(self):
        self.assertGreater(floq.linalg.operator_distance(u1, u2), 0.0)



class TestOperatorFidelityDeriv(CustomAssertions):

    def test_d_operator_fidelity(self):
        dus = np.array([u3, u4])
        target = np.array([0.302601, -0.291255])
        actual = floq.linalg.d_operator_fidelity(u1, dus, u2)
        self.assertArrayEqual(actual, target, 4)



class TestInnerProduct(CustomAssertions):
    def test_inner_product(self):
        result = floq.linalg.inner(v1, u1, v2)
        self.assertAlmostEqualWithDecimals(result, -0.362492 - 0.013523j, 4)



class TestHSProduct(CustomAssertions):
    def test_product(self):
        product = floq.linalg.hilbert_schmidt_product(u1, u2)
        self.assertAlmostEqualWithDecimals(product, 0.113672 + 0.830189j, 4)
