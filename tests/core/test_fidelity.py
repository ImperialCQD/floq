from tests.assertions import CustomAssertions
import numpy as np
import floq.fidelity as f

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
        fid = f.transfer_fidelity(u1, v2, v1)
        self.assertAlmostEqualWithDecimals(fid, 0.131584, 4)


class TestTransferDistance(CustomAssertions):

    def test_is_zero_if_identity(self):
        v = np.array([1.0, 1.0j])/1.41421
        fid = f.transfer_distance(np.eye(2), v, v)
        self.assertAlmostEqualWithDecimals(fid, 0.0, 4)

    def test_is_zero_if_bit_flip(self):
        zero = np.array([1.0, 0.0])
        one = np.array([0.0, 1.0])
        flip = np.array([[0, 1], [1, 0]])
        fid = f.transfer_distance(flip, zero, one)
        self.assertAlmostEqualWithDecimals(fid, 0.0, 4)



class TestOperatorFidelity(CustomAssertions):

    def test_operator_fidelity(self):
        fid = f.operator_fidelity(u1, u2)
        self.assertAlmostEqualWithDecimals(fid, 0.0378906, 4)



class TestOperatorDistance(CustomAssertions):

    def test_zero_if_equal(self):
        self.assertAlmostEqualWithDecimals(0.0, f.operator_distance(u1, u1))

    def test_pos_if_nonequal(self):
        self.assertGreater(f.operator_distance(u1, u2), 0.0)



class TestOperatorFidelityDeriv(CustomAssertions):

    def test_d_operator_fidelity(self):
        dus = np.array([u3, u4])
        target = np.array([0.302601, -0.291255])
        actual = f.d_operator_fidelity(u1, dus, u2)
        self.assertArrayEqual(actual, target, 4)



class TestInnerProduct(CustomAssertions):
    def test_inner_product(self):
        result = f.inner(v1, u1, v2)
        self.assertAlmostEqualWithDecimals(result, -0.362492 - 0.013523j, 4)



class TestHSProduct(CustomAssertions):
    def test_product(self):
        product = f.hilbert_schmidt_product(u1, u2)
        self.assertAlmostEqualWithDecimals(product, 0.113672 + 0.830189j, 4)
