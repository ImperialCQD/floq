from unittest import TestCase
import numpy as np
from floq.helpers.matrix import is_unitary, adjoint, gram_schmidt
from tests.assertions import CustomAssertions


class TestIsUnitary(TestCase):
    def test_true_if_unitary(self):
        u = np.array([[-0.288822 - 0.154483j, 0.20768 - 0.22441j, 0.0949032 - 0.0560178j, -0.385994 + 0.210021j, 0.423002 - 0.605778j, 0.135684 - 0.172261j], [0.0998628 - 0.364186j, 0.408817 - 0.35846j, -0.224508 - 0.550201j, 0.258427 + 0.263299j, -0.0297947 + 0.180679j, -0.0134853 + 0.197029j], [0.541087 - 0.216046j, -0.306777 + 0.0439077j, -0.479354 + 0.0395382j, -0.474755 + 0.264776j, -0.0971467 - 0.0167121j, 0.121192 - 0.115168j], [-0.0479833 - 0.133938j, 0.0696875 - 0.539678j, 0.314762 + 0.391157j, -0.376453 + 0.00569747j, -0.348676 + 0.2061j, 0.0588683 + 0.34972j], [-0.524482 + 0.213402j, 0.152127 + 0.111274j, -0.308402 - 0.134059j, -0.448647 + 0.120202j, -0.0680734 + 0.435883j, -0.295969 - 0.181141j], [-0.119405 + 0.235674j, 0.349453 + 0.247169j, -0.169971 + 0.0966179j, 0.0310919 + 0.129778j, -0.228356 + 0.00511762j, 0.793243 + 0.0977203j]])
        self.assertTrue(is_unitary(u, 1e-5))


    def test_true_if_not_unitary(self):
        u = np.array([[-5.288822 - 0.154483j, 0.20768 - 0.22441j, 0.0949032 - 0.0560178j, -0.385994 + 0.210021j, 0.423002 - 0.605778j, 0.135684 - 0.172261j], [0.0998628 - 0.364186j, 0.408817 - 0.35846j, -0.224508 - 0.550201j, 0.258427 + 0.263299j, -0.0297947 + 0.180679j, -0.0134853 + 0.197029j], [0.541087 - 0.216046j, -0.306777 + 0.0439077j, -0.479354 + 0.0395382j, -0.474755 + 0.264776j, -0.0971467 - 0.0167121j, 0.121192 - 0.115168j], [-0.0479833 - 0.133938j, 0.0696875 - 0.539678j, 0.314762 + 0.391157j, -0.376453 + 0.00569747j, -0.348676 + 0.2061j, 0.0588683 + 0.34972j], [-0.524482 + 0.213402j, 0.152127 + 0.111274j, -0.308402 - 0.134059j, -0.448647 + 0.120202j, -0.0680734 + 0.435883j, -0.295969 - 0.181141j], [-0.119405 + 0.235674j, 0.349453 + 0.247169j, -0.169971 + 0.0966179j, 0.0310919 + 0.129778j, -0.228356 + 0.00511762j, 0.793243 + 0.0977203j]])
        self.assertFalse(is_unitary(u))


class TestAdjoint(TestCase, CustomAssertions):

    def test_does_right_thing(self):
        u = np.array([[2.3+12j, -13j+5], [1j+12.1, 0.3j+0.1]])
        target = np.array([[2.3-12j, -1j+12.1], [+13j+5, -0.3j+0.1]])
        self.assertArrayEqual(adjoint(u), target)


class TestGramSchmidt(TestCase, CustomAssertions):

    def setUp(self):
        self.array = np.array([[1.0, 0.0, 3.0],
                               [2.0, 1.0, 2.0],
                               [3.0, 1.0, 1.0]])
        self.res = gram_schmidt(self.array)
        self.x = self.res[:, 0]
        self.y = self.res[:, 1]
        self.z = self.res[:, 2]

    def test_orthogonality_x_y(self):
        self.assertAlmostEqual(np.dot(self.x, self.y), 0.0)

    def test_orthogonality_x_z(self):
        self.assertAlmostEqual(np.dot(self.x, self.z), 0.0)

    def test_orthogonality_y_z(self):
        print self.y
        print self.z
        self.assertAlmostEqual(np.dot(self.y, self.z), 0.0)

    def test_normalised_x(self):
        self.assertAlmostEqual(np.sqrt(self.x.dot(self.x)), 1.0)

    def test_normalised_y(self):
        self.assertAlmostEqual(np.sqrt(self.y.dot(self.y)), 1.0)

    def test_normalised_z(self):
        self.assertAlmostEqual(np.sqrt(self.z.dot(self.z)), 1.0)
