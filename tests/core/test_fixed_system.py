from unittest import TestCase
import numpy as np
import floq.core.fixed_system as fs
import floq.errors as er
import floq.core.evolution
import tests.rabi as rabi
from tests.assertions import CustomAssertions
from mock import MagicMock


class TestFixedSystemInit(CustomAssertions):

    def setUp(self):
        self.hf = np.zeros([5, 10, 10])
        self.dhf = np.zeros([3, 10, 10])
        self.omega = 3.0
        self.t = 1.0
        self.nz = 9
        self.problem = fs.FixedSystem(self.hf, self.dhf, self.nz, self.omega, self.t)


    def test_set_hf(self):
        self.assertArrayEqual(self.problem.hf, self.hf)

    def test_set_dhf(self):
        self.assertArrayEqual(self.problem.dhf, self.dhf)

    def test_set_dim(self):
        self.assertEqual(self.problem.params.dim, 10)

    def test_set_nc(self):
        self.assertEqual(self.problem.params.nc, 5)

    def test_set_np(self):
        self.assertEqual(self.problem.params.np, 3)



class TestFixedSystemMaxNZ(TestCase):
    def setUp(self):
        g = 5.0
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])

        nz = 3
        dim = 2
        omega = 5.0
        t = 20.5
        self.s = fs.FixedSystem(hf, dhf, nz, omega, t)
        self.s.max_nz = 9


    def test_raise_MaxNZError(self):
        res = (self.s._test_nz())[1]

        mock = MagicMock(return_value=[False, res])  # make nz test fail
        self.s._test_nz = mock

        with self.assertRaises(er.NZTooLargeError):
            self.s.u()



class TestEvolveFixedSystem(CustomAssertions):
    def setUp(self):
        g = 5.0
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])

        nz = 3
        dim = 2
        omega = 5.0
        t = 20.5
        self.s = fs.FixedSystem(hf, dhf, nz, omega, t)

        self.u = rabi.u(g, e1, e2, omega, t)
        self.ucal = self.s.u

        self.um = np.matrix(self.ucal)

    def test_gives_unitary(self):
        uu = self.um*self.um.getH()
        identity = np.identity(2)
        self.assertArrayEqual(uu, identity, 8)

    def test_is_correct_u(self):
        self.assertArrayEqual(self.u, self.ucal, 8)



class TestEvolveFixedSystemWithDerivs(CustomAssertions):
    def setUp(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])

        nz = 3
        dim = 2
        omega = 5.0
        t = 1.5
        s = fs.FixedSystem(hf, dhf, nz, omega, t)

        self.ducal = np.array([[-0.43745 + 0.180865j, 0.092544 - 0.0993391j],
                            [-0.0611011 - 0.121241j, -0.36949-0.295891j]])
        self.du = s.du

    def test_is_correct_du(self):
        self.assertArrayEqual(self.ducal, self.du)



class TestFixedSystemCaching(TestCase):
    def setUp(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])

        nz = 15
        dim = 2
        omega = 5.0
        t = 20.5
        self.s = fs.FixedSystem(hf, dhf, nz, omega, t)

    def test_compute_u_once(self):
        u = self.s.u
        mock = MagicMock()
        self.s._compute_u = mock
        u = self.s.u
        mock.assert_not_called()

    def test_compute_udot_once(self):
        udot = self.s.udot
        mock = MagicMock()
        self.s._compute_udot = mock
        udot = self.s.udot
        mock.assert_not_called()


    def test_compute_du_once(self):
        du = self.s.du
        mock = MagicMock()
        self.s._compute_du = mock
        du = self.s.du
        mock.assert_not_called()



class TestFixedSystemParametersInit(TestCase):
    def setUp(self):
        self.dim = 2
        self.nz = 5
        self.nc = 3
        self.np = 3
        self.omega = 3.0
        self.t = 4.0
        self.decimals = 5

        self.p = fs.FixedSystemParameters(self.dim, self.nz, self.nc, self.np,
                                          self.omega, self.t, self.decimals)

    def test_raise_error_if_nz_even(self):
        with self.assertRaises(er.UsageError):
            fs.FixedSystemParameters(self.dim, self.nz*2, self.nc, self.np,
                                     self.omega, self.t, self.decimals)

    def test_raise_error_if_nc_even(self):
        with self.assertRaises(er.UsageError):
            fs.FixedSystemParameters(self.dim, self.nz, self.nc*2, self.np,
                                     self.omega, self.t, self.decimals)

    def test_set_k_dim(self):
        self.assertEqual(self.p.k_dim, self.dim*self.nz)

    def test_set_nz_max(self):
        self.assertEqual(self.p.nz_max, 2)

    def test_set_nz_min(self):
        self.assertEqual(self.p.nz_min, -2)
