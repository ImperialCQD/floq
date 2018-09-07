from tests.assertions import CustomAssertions
from tests.ryd import rydberg_atoms
import numpy as np
import floq
from . import rabi
from scipy.linalg import logm

# Test whether U and dU are computed correctly
# in a variety of cases

# Rabi problem

class TestRabiUfromFixedSystem(CustomAssertions):
    def setUp(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)

        nz = 11
        dim = 2
        omega = 5.0
        t = 20.5
        p = floq.core.fixed_system.FixedSystemParameters(dim=dim, nz=nz,
                                                         omega=omega, t=t, nc=3,
                                                         np=1)
        self.u = rabi.u(g, e1, e2, omega, t)
        self.ucal = floq.core.evolution.get_u(hf, p)
        self.um = np.matrix(self.ucal)

    def test_gives_unitary(self):
        uu = self.um*self.um.getH()
        identity = np.identity(2)
        self.assertArrayEqual(uu, identity, 8)

    def test_is_correct_u(self):
        self.assertArrayEqual(self.u, self.ucal, 8)


class TestRabidUfromFixedSystem(CustomAssertions):
    def setUp(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])
        nz = 21
        dim = 2
        omega = 5.0
        t = 1.5
        p = floq.core.fixed_system.FixedSystemParameters(dim, nz, nc=3,
                                                         omega=omega, t=t, np=1)
        k = floq.core.evolution.assemble_k(hf, p)
        vals, vecs = floq.core.evolution.find_eigensystem(k, p)
        psi = floq.core.evolution.calculate_psi(vecs, p)
        self.du = np.array([[-0.43745 + 0.180865j, 0.092544 - 0.0993391j],
                            [-0.0611011 - 0.121241j, -0.36949 - 0.295891j]])
        dk = floq.core.evolution.assemble_dk(dhf, p)
        self.ducal = floq.core.evolution.calculate_du(dk, psi, vals, vecs, p)

    def test_is_correct_du(self):
        self.assertArrayEqual(self.ducal, self.du)

class TestRabiUandDUfromFixedSystem(CustomAssertions):
    def setUp(self):
        g = 0.5
        e1 = 1.2
        e2 = 2.8
        hf = rabi.hf(g, e1, e2)
        dhf = np.array([rabi.hf(1.0, 0, 0)])
        nz = 21
        dim = 2
        omega = 5.0
        t = 1.5
        p = floq.core.fixed_system.FixedSystemParameters(dim=dim, nz=nz,
                                                         omega=omega, t=t, nc=3,
                                                         np=1)
        self.du = np.array([[-0.43745 + 0.180865j, 0.092544 - 0.0993391j],
                            [-0.0611011 - 0.121241j, -0.36949-0.295891j]])
        self.ucal, self.ducal = floq.core.evolution.get_u_and_du(hf, dhf, p)

    def test_is_correct_du(self):
        self.assertArrayEqual(self.ducal, self.du)

# NV Centre Spin

class TestSpinUfromSpinSystem(CustomAssertions):
    def test_spin_u_correct(self):
        target = np.array([[0.105818 - 0.324164j, -0.601164 - 0.722718j],
                           [0.601164 - 0.722718j, 0.105818 + 0.324164j]])

        spin = floq.systems.spin(2, 1.0, 1.1, omega=1.5)
        controls = np.array([1.5, 1.5, 1.5, 1.5])
        result = spin.u(1.0, controls)
        self.assertArrayEqual(target, result)

class TestSpinUfromFixedSystem(CustomAssertions):
    def setUp(self):
        controls = np.array([1.2, 1.3, 4.5, 3.3, -0.8, 0.9, 3.98, -4.0, 0.9, 1.0])
        hf = floq.systems._spin_hamiltonian(5, 0.1, controls)
        nz = 51
        dim = 2
        omega = 1.3
        t = 0.6
        p = floq.core.fixed_system.FixedSystemParameters(dim=dim, nz=nz,
                                                         omega=omega, t=t,
                                                         nc=11, np=10)
        self.u = np.array([[-0.150824 + 0.220144j, -0.132296 - 0.954613j],
                           [0.132296 - 0.954613j, -0.150824 - 0.220144j]])
        self.ucal = floq.core.evolution.get_u(hf, p)
        self.um = np.matrix(self.ucal)

    def test_gives_unitary(self):
        uu = self.um*self.um.getH()
        identity = np.identity(2)
        self.assertArrayEqual(uu, identity, 8)

    def test_is_correct_u(self):
        self.assertArrayEqual(self.u, self.ucal, 8)


# Rydberg Atoms
class TestRydbergAtoms(CustomAssertions):
    def setUp(self):
        self.rvec = np.array([0.5, 0.5, 0.7071067811865476])
        self.rvec2 = np.array([3.0, 2.0, 2.0])
        self.mu = 1.2
        self.delta = 0.2
        self.omega = 2.1

    def test_h(self):
        target = np.array([[[0, 0, 0. - 2.30318j, 0. + 5.75795j, 0, 0, 0, 0, 0. - 2.30318j, 0, 0, 0, 0. + 5.75795j, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 2.30318j, 0, 0, 0, 0. + 5.75795j, 0, 0], [0. - 2.30318j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 2.30318j, 0, 0, 0, 0. + 5.75795j, 0], [0. + 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 2.30318j, 0, 0, 0, 0. + 5.75795j], [0, 0, 0, 0, 0, 0, 0. - 2.30318j, 0. + 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0. - 2.30318j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0. + 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0. - 2.30318j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 2.30318j, 0. + 5.75795j, 0, 0, 0, 0], [0, 0. - 2.30318j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0. - 2.30318j, 0, 0, 0, 0, 0, 0. - 2.30318j, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0. - 2.30318j, 0, 0, 0, 0, 0. + 5.75795j, 0, 0, 0, 0, 0, 0, 0], [0. + 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 2.30318j, 0. + 5.75795j], [0, 0. + 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0. + 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 2.30318j, 0, 0, 0], [0, 0, 0, 0. + 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 5.75795j, 0, 0, 0]], [[0, 0, 0. - 1.91932j, 0. + 4.2225j, 0, 0, 0, 0, 0. - 1.91932j, 0, 0, 0, 0. + 4.2225j, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 1.91932j, 0, 0, 0, 0. + 4.2225j, 0, 0], [0. - 1.91932j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 1.91932j, 0, 0, 0, 0. + 4.2225j, 0], [0. + 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 1.91932j, 0, 0, 0, 0. + 4.2225j], [0, 0, 0, 0, 0, 0, 0. - 1.91932j, 0. + 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0. - 1.91932j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0. + 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0. - 1.91932j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 1.91932j, 0. + 4.2225j, 0, 0, 0, 0], [0, 0. - 1.91932j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0. - 1.91932j, 0, 0, 0, 0, 0, 0. - 1.91932j, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0. - 1.91932j, 0, 0, 0, 0, 0. + 4.2225j, 0, 0, 0, 0, 0, 0, 0], [0. + 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 1.91932j, 0. + 4.2225j], [0, 0. + 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0. + 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 1.91932j, 0, 0, 0], [0, 0, 0, 0. + 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 4.2225j, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, -0.2, 0, 0, 351.003, 0, 0, 0, -1053.01 - 1053.01j, 0, 0, 0, 0. + 1053.01j, 0, 0, 0], [0, 0, -0.2, 0, -1053.01 + 1053.01j, 0, 0, 0, -702.006, 0, 0, 0, 1053.01 + 1053.01j, 0, 0, 0], [0, 0, 0, -0.2, 0. - 1053.01j, 0, 0, 0, 1053.01 - 1053.01j, 0, 0, 0, 351.003, 0, 0, 0], [0, 351.003, -1053.01 - 1053.01j, 0. + 1053.01j, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -0.4, 0, 0, 0, 0, 0, 0, 0, 0], [0, -1053.01 + 1053.01j, -702.006, 1053.01 + 1053.01j, 0, 0, 0, 0, -0.2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4, 0, 0, 0, 0], [0, 0. - 1053.01j, 1053.01 - 1053.01j, 351.003, 0, 0, 0, 0, 0, 0, 0, 0, -0.2, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.4]], [[0, 0, 0. + 1.91932j, 0. - 4.2225j, 0, 0, 0, 0, 0. + 1.91932j, 0, 0, 0, 0. - 4.2225j, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 1.91932j, 0, 0, 0, 0. - 4.2225j, 0, 0], [0. + 1.91932j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 1.91932j, 0, 0, 0, 0. - 4.2225j, 0], [0. - 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 1.91932j, 0, 0, 0, 0. - 4.2225j], [0, 0, 0, 0, 0, 0, 0. + 1.91932j, 0. - 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0. + 1.91932j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0. - 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0. + 1.91932j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 1.91932j, 0. - 4.2225j, 0, 0, 0, 0], [0, 0. + 1.91932j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0. + 1.91932j, 0, 0, 0, 0, 0, 0. + 1.91932j, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0. + 1.91932j, 0, 0, 0, 0, 0. - 4.2225j, 0, 0, 0, 0, 0, 0, 0], [0. - 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 1.91932j, 0. - 4.2225j], [0, 0. - 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0. - 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 1.91932j, 0, 0, 0], [0, 0, 0, 0. - 4.2225j, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 4.2225j, 0, 0, 0]], [[0, 0, 0. + 2.30318j, 0. - 5.75795j, 0, 0, 0, 0, 0. + 2.30318j, 0, 0, 0, 0. - 5.75795j, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 2.30318j, 0, 0, 0, 0. - 5.75795j, 0, 0], [0. + 2.30318j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 2.30318j, 0, 0, 0, 0. - 5.75795j, 0], [0. - 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 2.30318j, 0, 0, 0, 0. - 5.75795j], [0, 0, 0, 0, 0, 0, 0. + 2.30318j, 0. - 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0. + 2.30318j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0. - 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0. + 2.30318j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 2.30318j, 0. - 5.75795j, 0, 0, 0, 0], [0, 0. + 2.30318j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0. + 2.30318j, 0, 0, 0, 0, 0, 0. + 2.30318j, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0. + 2.30318j, 0, 0, 0, 0, 0. - 5.75795j, 0, 0, 0, 0, 0, 0, 0], [0. - 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 2.30318j, 0. - 5.75795j], [0, 0. - 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0. - 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0. + 2.30318j, 0, 0, 0], [0, 0, 0, 0. - 5.75795j, 0, 0, 0, 0, 0, 0, 0, 0, 0. - 5.75795j, 0, 0, 0]]])
        controls = np.array([0.5, 1.1, 0.6, 1.5])
        system = rydberg_atoms(2, self.rvec, self.mu, self.delta, self.omega)
        self.assertArrayEqual(target, system.hamiltonian(controls))

    def test_u(self):
        target = np.array([[0.14099 + 0.243431j, -0.235477 - 0.175999j, -0.073094 - 0.0604126j, -0.139088 - 0.110263j, -0.235477 - 0.175999j, 0. + 0.j, -0.0288308 - 0.0187422j, 0.0457725 + 0.00437857j, -0.073094 - 0.0604126j, -0.0288308 - 0.0187422j, -0.140161 + 0.0282982j, 0.289831 - 0.0548884j, -0.139088 - 0.110263j, 0.0457725 + 0.00437857j, 0.289831 - 0.0548884j, -0.682265 + 0.0215671j], [0.0249514 + 0.0315845j, 0.476767 - 0.121017j, 0.189515 - 0.0955939j, -0.15735 + 0.0201964j, -0.289881 + 0.0839668j, 0. + 0.j, 0.0997704 + 0.119899j, -0.235431 - 0.296563j, 0.128208 - 0.445984j, 0.0820874 + 0.000202917j, 0.119303 + 0.024769j, -0.113272 - 0.201712j, -0.223447 + 0.0574001j, -0.213215 + 0.042095j, -0.0941613 + 0.126554j, -0.107851 - 0.00210075j], [-0.0575678 - 0.0854277j, 0.0139823 + 0.0410847j, 0.346227 + 0.0271266j, -0.192916 + 0.172734j, -0.102643 - 0.32196j, 0. + 0.j, 0.0918238 + 0.0751787j, -0.212264 - 0.189556j, -0.219589 + 0.549339j, 0.00827412 - 0.0882803j, -0.059485 + 0.0688941j, 0.0171301 + 0.0220121j, -0.153039 + 0.347093j, -0.0540536 + 0.21235j, 0.0334496 - 0.0740452j, -0.00326951 - 0.158549j], [0.0947111 + 0.302302j, 0.244803 - 0.348978j, 0.0095817 - 0.0100746j, 0.554975 - 0.241828j, 0.0580354 - 0.14061j, 0. + 0.j, 0.0142677 - 0.0179474j, 0.0219676 + 0.0713832j, -0.0154937 + 0.233219j, 0.13488 + 0.0356895j, 0.0326452 + 0.0880699j, 0.0335666 - 0.0124915j, -0.179493 + 0.0992966j, -0.245811 - 0.122055j, -0.10416 - 0.258571j, 0.0495446 + 0.200627j], [0.0249514 + 0.0315845j, -0.289881 + 0.0839668j, 0.128208 - 0.445984j, -0.223447 + 0.0574001j, 0.476767 - 0.121017j, 0. + 0.j, 0.0820874 + 0.000202917j, -0.213215 + 0.042095j, 0.189515 - 0.0955939j, 0.0997704 + 0.119899j, 0.119303 + 0.024769j, -0.0941613 + 0.126554j, -0.15735 + 0.0201964j, -0.235431 - 0.296563j, -0.113272 - 0.201712j, -0.107851 - 0.00210075j], [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0.935897 + 0.352274j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j], [-0.0368925 + 0.00511545j, -0.0345354 + 0.0214151j, 0.0117757 + 0.0182706j, 0.014589 + 0.0216928j, 0.074563 + 0.110577j, 0. + 0.j, 0.837343 + 0.302371j, 0.255867 + 0.117213j, 0.090146 + 0.0502756j, 0.00626085 - 0.0561297j, -0.0345116 - 0.0387912j, 0.112221 + 0.0119532j, -0.114365 - 0.146233j, -0.04169 + 0.191189j, -0.00882616 + 0.0283131j, -0.00827915 + 0.057106j], [0.0780155 - 0.0374003j, -0.00903028 - 0.0231809j, -0.010063 - 0.00264291j, -0.026557 - 0.0647374j, -0.256817 - 0.251457j, 0. + 0.j, 0.259505 + 0.113206j, 0.241697 + 0.0776364j, -0.223527 - 0.136334j, -0.0467851 + 0.178175j, 0.091025 + 0.0574368j, -0.266653 - 0.0204111j, 0.215913 + 0.361744j, 0.171078 - 0.551337j, 0.021576 + 0.00543457j, 0.000507651 - 0.127664j], [-0.0575678 - 0.0854277j, -0.102643 - 0.32196j, -0.219589 + 0.549339j, -0.153039 + 0.347093j, 0.0139823 + 0.0410847j, 0. + 0.j, 0.00827412 - 0.0882803j, -0.0540536 + 0.21235j, 0.346227 + 0.0271266j, 0.0918238 + 0.0751787j, -0.059485 + 0.0688941j, 0.0334496 - 0.0740452j, -0.192916 + 0.172734j, -0.212264 - 0.189556j, 0.0171301 + 0.0220121j, -0.00326951 - 0.158549j], [-0.0368925 + 0.00511545j, 0.074563 + 0.110577j, 0.090146 + 0.0502756j, -0.114365 - 0.146233j, -0.0345354 + 0.0214151j, 0. + 0.j, 0.00626085 - 0.0561297j, -0.04169 + 0.191189j, 0.0117757 + 0.0182706j, 0.837343 + 0.302371j, -0.0345116 - 0.0387912j, -0.00882616 + 0.0283131j, 0.014589 + 0.0216928j, 0.255867 + 0.117213j, 0.112221 + 0.0119532j, -0.00827915 + 0.057106j], [-0.159499 + 0.0578669j, 0.0284379 - 0.00713525j, -0.0557475 + 0.0969734j, -0.103326 - 0.0355881j, 0.0284379 - 0.00713525j, 0. + 0.j, -0.0310252 - 0.0394194j, 0.0812356 + 0.0608832j, -0.0557475 + 0.0969734j, -0.0310252 - 0.0394194j, 0.857853 + 0.304607j, 0.0746555 + 0.142761j, -0.103326 - 0.0355881j, 0.0812356 + 0.0608832j, 0.0746555 + 0.142761j, -0.0838291 + 0.0854751j], [0.311118 - 0.0821114j, 0.0669511 + 0.0279554j, 0.0761656 + 0.0934688j, 0.0204188 + 0.0361347j, -0.0365434 + 0.0775406j, 0. + 0.j, 0.0999277 + 0.00514494j, -0.238797 - 0.00743556j, -0.0772191 - 0.262424j, -0.024219 + 0.0447513j, 0.080541 + 0.142616j, 0.622012 + 0.250005j, 0.158425 + 0.0796654j, 0.0676581 - 0.0321745j, 0.0482904 - 0.362398j, 0.218723 - 0.168198j], [0.0947111 + 0.302302j, 0.0580354 - 0.14061j, -0.0154937 + 0.233219j, -0.179493 + 0.0992966j, 0.244803 - 0.348978j, 0. + 0.j, 0.13488 + 0.0356895j, -0.245811 - 0.122055j, 0.0095817 - 0.0100746j, 0.0142677 - 0.0179474j, 0.0326452 + 0.0880699j, -0.10416 - 0.258571j, 0.554975 - 0.241828j, 0.0219676 + 0.0713832j, 0.0335666 - 0.0124915j, 0.0495446 + 0.200627j], [0.0780155 - 0.0374003j, -0.256817 - 0.251457j, -0.223527 - 0.136334j, 0.215913 + 0.361744j, -0.00903028 - 0.0231809j, 0. + 0.j, -0.0467851 + 0.178175j, 0.171078 - 0.551337j, -0.010063 - 0.00264291j, 0.259505 + 0.113206j, 0.091025 + 0.0574368j, 0.021576 + 0.00543457j, -0.026557 - 0.0647374j, 0.241697 + 0.0776364j, -0.266653 - 0.0204111j, 0.000507651 - 0.127664j], [0.311118 - 0.0821114j, -0.0365434 + 0.0775406j, -0.0772191 - 0.262424j, 0.158425 + 0.0796654j, 0.0669511 + 0.0279554j, 0. + 0.j, -0.024219 + 0.0447513j, 0.0676581 - 0.0321745j, 0.0761656 + 0.0934688j, 0.0999277 + 0.00514494j, 0.080541 + 0.142616j, 0.0482904 - 0.362398j, 0.0204188 + 0.0361347j, -0.238797 - 0.00743556j, 0.622012 + 0.250005j, 0.218723 - 0.168198j], [-0.664798 + 0.0110052j, -0.27059 - 0.153875j, -0.0208902 - 0.151942j, -0.0288611 - 0.084489j, -0.27059 - 0.153875j, 0. + 0.j, 0.0191762 + 0.0217013j, -0.0771723 - 0.0569021j, -0.0208902 - 0.151942j, 0.0191762 + 0.0217013j, -0.0462297 + 0.0509861j, 0.185455 - 0.138449j, -0.0288611 - 0.084489j, -0.0771723 - 0.0569021j, 0.185455 - 0.138449j, 0.38529 + 0.144111j]])
        controls = np.array([0.5, 1.1, 0.6, 1.5])
        system = rydberg_atoms(2, self.rvec2, self.mu, self.delta, self.omega)
        self.assertArrayEqual(target, system.u(0.9, controls))

    def test_du_dt(self):
        target = np.array([[-11.2843 + 3.96466j, 6.21544 + 5.674j, 0.29355 - 0.981686j, 6.00339 + 8.68081j, 6.21544 + 5.674j, 0. + 0.j, -0.387437 + 1.79209j, 1.00461 - 1.88023j, 0.29355 - 0.981686j, -0.387437 + 1.79209j, -1.97987 + 1.91933j, 4.16232 - 1.531j, 6.00339 + 8.68081j, 1.00461 - 1.88023j, 4.16232 - 1.531j, -8.90917 + 1.69959j], [7.09722 + 7.20337j, 10.5886 - 7.38135j, -0.351365 - 7.64688j, -13.8361 + 1.63907j, 2.22836 - 8.32953j, 0. + 0.j, 1.02671 + 0.00865642j, 1.53669 - 0.878298j, -3.21195 + 4.9568j, -2.03677 - 1.07615j, -0.275171 + 2.13398j, -3.94375 - 3.19977j, 10.1329 - 15.7069j, 6.44813 + 1.54683j, 1.09331 - 6.11132j, 7.76999 + 5.08031j], [8.24127 + 7.94294j, -10.1622 - 4.34337j, 19.1797 + 0.999862j, -3.7364 + 6.10518j, 2.35876 + 1.23786j, 0. + 0.j, 1.01366 - 0.450349j, -4.76988 + 0.209596j, 2.01759 + 1.72798j, -0.957971 + 3.84787j, 2.80045 - 0.892484j, -1.53621 - 3.09844j, -0.625756 - 9.76217j, 2.80393 - 9.74402j, -1.37572 + 3.40528j, 5.10916 + 8.63562j], [-6.72677 - 9.78938j, 9.27373 - 3.15607j, 7.28549 + 3.98293j, 15.2404 - 3.78141j, -3.60128 - 9.72781j, 0. + 0.j, -3.26091 - 0.428822j, 9.78922 + 0.0998382j, -6.22027 - 6.11656j, -0.538779 - 5.47667j, -1.28989 - 4.94697j, 7.40529 + 1.68516j, 12.534 - 0.180749j, 2.11855 + 12.4685j, 1.56839 + 12.4785j, -4.64588 - 3.69552j], [7.09722 + 7.20337j, 2.22836 - 8.32953j, -3.21195 + 4.9568j, 10.1329 - 15.7069j, 10.5886 - 7.38135j, 0. + 0.j, -2.03677 - 1.07615j, 6.44813 + 1.54683j, -0.351365 - 7.64688j, 1.02671 + 0.00865642j, -0.275171 + 2.13398j, 1.09331 - 6.11132j, -13.8361 + 1.63907j, 1.53669 - 0.878298j, -3.94375 - 3.19977j, 7.76999 + 5.08031j], [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, -0.14091 + 0.374359j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j], [0.218193 - 0.188722j, 0.576763 + 2.00708j, -3.11647 - 0.889082j, 0.391386 + 1.56355j, -0.887965 - 3.29401j, 0. + 0.j, -0.119534 - 0.237326j, 0.246574 + 1.58874j, -0.686583 - 1.28515j, 0.858327 - 0.693052j, 0.188205 - 0.845528j, 0.877465 + 0.701322j, 0.199278 + 1.05127j, -2.14398 + 1.62463j, -1.41759 + 0.786164j, -0.0374522 + 0.748555j], [-0.512449 + 0.447804j, -1.39242 - 4.84308j, 7.4466 + 2.13635j, -0.932139 - 3.74089j, 2.12108 + 7.85689j, 0. + 0.j, -0.0486686 + 1.4742j, -0.733806 - 3.46281j, 1.65054 + 3.0745j, -2.07295 + 1.64694j, -0.436514 + 2.02815j, -2.10456 - 1.67863j, -0.481836 - 2.54067j, 5.17159 - 3.86201j, 3.36542 - 1.88246j, 0.0860508 - 1.8003j], [8.24127 + 7.94294j, 2.35876 + 1.23786j, 2.01759 + 1.72798j, -0.625756 - 9.76217j, -10.1622 - 4.34337j, 0. + 0.j, -0.957971 + 3.84787j, 2.80393 - 9.74402j, 19.1797 + 0.999862j, 1.01366 - 0.450349j, 2.80045 - 0.892484j, -1.37572 + 3.40528j, -3.7364 + 6.10518j, -4.76988 + 0.209596j, -1.53621 - 3.09844j, 5.10916 + 8.63562j], [0.218193 - 0.188722j, -0.887965 - 3.29401j, -0.686583 - 1.28515j, 0.199278 + 1.05127j, 0.576763 + 2.00708j, 0. + 0.j, 0.858327 - 0.693052j, -2.14398 + 1.62463j, -3.11647 - 0.889082j, -0.119534 - 0.237326j, 0.188205 - 0.845528j, -1.41759 + 0.786164j, 0.391386 + 1.56355j, 0.246574 + 1.58874j, 0.877465 + 0.701322j, -0.0374522 + 0.748555j], [-1.21415 + 0.738789j, -1.95532 + 0.629503j, 3.98007 - 0.905251j, 3.63816 + 2.37053j, -1.95532 + 0.629503j, 0. + 0.j, -0.0755532 - 0.71025j, 0.134498 + 1.88914j, 3.98007 - 0.905251j, -0.0755532 - 0.71025j, 0.838759 + 1.17253j, -0.419887 - 0.322749j, 3.63816 + 2.37053j, 0.134498 + 1.88914j, -0.419887 - 0.322749j, -2.24474 + 0.0119897j], [3.56631 - 1.49675j, 2.93098 - 3.39348j, -9.27874 - 3.70231j, -7.49484 - 6.41591j, -1.69711 - 0.185865j, 0. + 0.j, 1.34659 + 0.0786429j, -3.04438 - 1.15109j, 1.27796 + 5.85754j, -1.02421 + 0.582998j, -0.593275 - 1.18843j, 1.04914 + 0.573168j, -2.22337 - 1.9061j, 2.32664 - 1.80303j, -2.02505 + 1.03143j, 4.11272 - 0.312373j], [-6.72677 - 9.78938j, -3.60128 - 9.72781j, -6.22027 - 6.11656j, 12.534 - 0.180749j, 9.27373 - 3.15607j, 0. + 0.j, -0.538779 - 5.47667j, 2.11855 + 12.4685j, 7.28549 + 3.98293j, -3.26091 - 0.428822j, -1.28989 - 4.94697j, 1.56839 + 12.4785j, 15.2404 - 3.78141j, 9.78922 + 0.0998382j, 7.40529 + 1.68516j, -4.64588 - 3.69552j], [-0.512449 + 0.447804j, 2.12108 + 7.85689j, 1.65054 + 3.0745j, -0.481836 - 2.54067j, -1.39242 - 4.84308j, 0. + 0.j, -2.07295 + 1.64694j, 5.17159 - 3.86201j, 7.4466 + 2.13635j, -0.0486686 + 1.4742j, -0.436514 + 2.02815j, 3.36542 - 1.88246j, -0.932139 - 3.74089j, -0.733806 - 3.46281j, -2.10456 - 1.67863j, 0.0860508 - 1.8003j], [3.56631 - 1.49675j, -1.69711 - 0.185865j, 1.27796 + 5.85754j, -2.22337 - 1.9061j, 2.93098 - 3.39348j, 0. + 0.j, -1.02421 + 0.582998j, 2.32664 - 1.80303j, -9.27874 - 3.70231j, 1.34659 + 0.0786429j, -0.593275 - 1.18843j, -2.02505 + 1.03143j, -7.49484 - 6.41591j, -3.04438 - 1.15109j, 1.04914 + 0.573168j, 4.11272 - 0.312373j], [-10.0977 + 2.89615j, 8.23517 + 4.94763j, -3.66453 - 0.10709j, 2.41326 + 6.25741j, 8.23517 + 4.94763j, 0. + 0.j, -0.304796 + 2.4976j, 0.868517 - 3.76775j, -3.66453 - 0.10709j, -0.304796 + 2.4976j, -2.96087 + 1.07145j, 4.58048 - 1.10421j, 2.41326 + 6.25741j, 0.868517 - 3.76775j, 4.58048 - 1.10421j, -6.75626 + 1.80818j]])

        controls = np.array([0.5, 1.1, 0.6, 1.5])
        system = rydberg_atoms(2, self.rvec2, self.mu, self.delta, self.omega)
        self.assertArrayEqual(target, system.du_dt(0.9, controls), decimals=3)

    def test_u_small_controls(self):
        target = np.array([[0.9969222454477519 - 0.029231983008565866*1j, 0.0018521709019542658 - 0.013871991035677534*1j, -0.013701661352071974 + 0.020898309414631677*1j, 0.0074296056411065495 - 0.025521108027416178*1j, 0.0018521709019542467 - 0.0138719910356775*1j, 0. + 0.*1j, -0.010801515522138046 - 0.01171372813930524*1j, 0.010801515522138046 + 0.011713728139305237*1j, -0.013701661352071876 + 0.02089830941463167*1j, -0.010801515522138046 - 0.011713728139305234*1j, -0.007484911771980837 - 0.014147324138826365*1j, -0.0011264886689647292 + 0.016277320198555403*1j, 0.007429605641106544 - 0.025521108027416178*1j, 0.010801515522138056 + 0.011713728139305216*1j, -0.001126488668964734 + 0.01627732019855541*1j, 0.00973788910991029 - 0.01840731625828446*1j], [0.029904665638181442 + 0.0023259333181533046*1j, 0.41511909262621827 + 0.07272769522448783*1j, -0.15574489901433816 - 0.14768724530362426*1j, 0.06035154730245276 + 0.26119554874054896*1j, 0.00739966020244234 + 0.06600095575619935*1j, 0. + 0.*1j, 0.00009196188450197134 + 0.004529020626193024*1j, -0.0000919618845019065 - 0.004529020626193017*1j, 0.3652926461086209 - 0.3662716721423586*1j, -0.007405763424798452 + 0.02642209225908369*1j, 0.012011475770630121 + 0.00011660497701472392*1j, -0.016527800359010053 - 0.004359533185051595*1j, -0.6431026286818122 + 0.12533135882419863*1j, 0.007405763424798441 - 0.026422092259083582*1j, -0.007433828316292848 - 0.0038088001191358112*1j, 0.011950152904672757 + 0.008051728327172804*1j], [0.00926723446917095 + 0.022689211784983197*1j, -0.1908425330321768 + 0.08440159825568128*1j, 0.5150223703611576 + 0.09137650062251415*1j, 0.1513104203576158 + 0.14908654542067248*1j, -0.19146567149904675 - 0.4778077618120901*1j, 0. + 0.*1j, 0.007840643427209488 - 0.0021783078596058617*1j, -0.007840643427209476 + 0.0021783078596058583*1j, -0.0466159495414713 + 0.3128765449183174*1j, -0.002784151386740268 - 0.008780582247955653*1j, -0.011582066123312562 + 0.032413135579274166*1j, -0.003627278016138079 + 0.00379292397864606*1j, -0.3694503332101318 + 0.36417886073480965*1j, 0.002784151386740263 + 0.008780582247955618*1j, 0.002941504971620984 - 0.03630639178961601*1j, 0.012267839167829737 + 0.00010033223169573168*1j], [-0.002384739076203496 - 0.007474120643855296*1j, 0.14618340557564807 - 0.22578614067378983*1j, 0.1915992829162693 - 0.08224803307094446*1j, 0.41373844537289767 + 0.0751036791680455*1j, 0.5438046323966192 + 0.3725087702082355*1j, 0. + 0.*1j, -0.007178724114563586 + 0.008022024485799339*1j, 0.00717872411456359 - 0.00802202448579938*1j, 0.1916710720890569 + 0.47889766775016646*1j, 0.008787246909836875 + 0.0072435488368881714*1j, -0.005112895313638854 + 0.011197038862113415*1j, 0.0008245494782927217 + 0.023658985286268277*1j, 0.004286152226219163 + 0.06736278925890729*1j, -0.008787246909836866 - 0.007243548836888163*1j, -0.0025772448515652994 - 0.0050271361699633695*1j, 0.006865590686911392 - 0.029828887978418383*1j], [0.029904665638181532 + 0.0023259333181533*1j, 0.007399660202442435 + 0.06600095575619876*1j, 0.36529264610862106 - 0.3662716721423583*1j, -0.6431026286818138 + 0.12533135882419869*1j, 0.4151190926262186 + 0.07272769522448777*1j, 0. + 0.*1j, -0.007405763424798442 + 0.026422092259083648*1j, 0.007405763424798442 - 0.02642209225908364*1j, -0.1557448990143396 - 0.14768724530362534*1j, 0.00009196188450197025 + 0.004529020626193058*1j, 0.012011475770630118 + 0.00011660497701471091*1j, -0.007433828316292913 - 0.00380880011913579*1j, 0.060351547302452506 + 0.2611955487405481*1j, -0.00009196188450191192 - 0.004529020626193039*1j, -0.016527800359009984 - 0.004359533185051586*1j, 0.011950152904672773 + 0.008051728327172774*1j], [0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0.935896823657306 + 0.35227423326732843*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j, 0. + 0.*1j], [0.013854240609468733 - 0.007346766056902616*1j, -0.0100846593410889 - 0.001247725648993535*1j, 0.006925061112831577 + 0.007209360953522757*1j, -0.002496558257397089 - 0.012174408594576774*1j, -0.0072533877120846465 + 0.026398231595981116*1j, 0. + 0.*1j, 0.935452290317501 + 0.35101077394732033*1j, 0.0004445333398057133 + 0.0012634593200081495*1j, 0.00504239728240438 - 0.0006799644704053819*1j, 0.0010674253644799771 - 0.003868118100378991*1j, 0.00556518910749318 - 0.004171733577648478*1j, -0.009096145835382038 + 0.00027257219116138756*1j, -0.006355435436167555 - 0.002451959174577355*1j, -0.001067425364479978 + 0.003868118100378989*1j, -0.005481323005962604 + 0.004302387118676581*1j, 0.009012279733851462 - 0.00040322573218949057*1j], [-0.013854240609468733 + 0.007346766056902616*1j, 0.0100846593410889 + 0.001247725648993535*1j, -0.006925061112831577 - 0.007209360953522757*1j, 0.002496558257397089 + 0.012174408594576774*1j, 0.0072533877120846465 - 0.026398231595981116*1j, 0. + 0.*1j, 0.0004445333398057171 + 0.0012634593200081539*1j, 0.9354522903175009 + 0.3510107739473205*1j, -0.00504239728240438 + 0.0006799644704053819*1j, -0.0010674253644799771 + 0.003868118100378991*1j, -0.00556518910749318 + 0.004171733577648478*1j, 0.009096145835382038 - 0.00027257219116138756*1j, 0.006355435436167555 + 0.002451959174577355*1j, 0.001067425364479978 - 0.003868118100378989*1j, 0.005481323005962604 - 0.004302387118676581*1j, -0.009012279733851462 + 0.00040322573218949057*1j], [0.009267234469170993 + 0.02268921178498316*1j, -0.19146567149904675 - 0.4778077618120902*1j, -0.046615949541471136 + 0.3128765449183172*1j, -0.36945033321013215 + 0.36417886073481043*1j, -0.19084253303217688 + 0.0844015982556813*1j, 0. + 0.*1j, -0.0027841513867402734 - 0.008780582247955658*1j, 0.0027841513867402764 + 0.008780582247955642*1j, 0.5150223703611586 + 0.09137650062251398*1j, 0.007840643427209415 - 0.002178307859605857*1j, -0.011582066123312617 + 0.032413135579274194*1j, 0.002941504971620973 - 0.03630639178961595*1j, 0.1513104203576158 + 0.14908654542067207*1j, -0.007840643427209464 + 0.002178307859605862*1j, -0.003627278016138047 + 0.0037929239786460523*1j, 0.012267839167829725 + 0.00010033223169574794*1j], [0.01385424060946873 - 0.007346766056902622*1j, -0.007253387712084646 + 0.026398231595981106*1j, 0.005042397282404355 - 0.0006799644704053858*1j, -0.0063554354361674455 - 0.0024519591745773566*1j, -0.010084659341088932 - 0.0012477256489935814*1j, 0. + 0.*1j, 0.0010674253644799747 - 0.0038681181003789892*1j, -0.0010674253644799771 + 0.003868118100378991*1j, 0.006925061112831578 + 0.007209360953522748*1j, 0.9354522903175008 + 0.35101077394732055*1j, 0.00556518910749318 - 0.004171733577648479*1j, -0.005481323005962598 + 0.004302387118676577*1j, -0.002496558257397097 - 0.01217440859457674*1j, 0.000444533339805713 + 0.0012634593200081521*1j, -0.00909614583538204 + 0.000272572191161386*1j, 0.009012279733851459 - 0.00040322573218949084*1j], [0.008700791270016056 - 0.011207350138679848*1j, 0.008376788418389026 - 0.008198330522067844*1j, -0.02310833097492097 + 0.02618218496372692*1j, -0.01142515755149116 - 0.005217816119632546*1j, 0.008376788418388946 - 0.008198330522067865*1j, 0. + 0.*1j, -0.0016275261970184055 - 0.0068885001107619895*1j, 0.0016275261970184053 + 0.006888500110761994*1j, -0.02310833097492103 + 0.02618218496372695*1j, -0.0016275261970184064 - 0.006888500110761996*1j, 0.9350923944607211 + 0.3490718470372944*1j, -0.0053209641109137305 + 0.005704141729771258*1j, -0.01142515755149117 - 0.005217816119632538*1j, 0.0016275261970184103 + 0.00688850011076199*1j, -0.00532096411091373 + 0.005704141729771269*1j, 0.011446357418413291 - 0.008205897229507873*1j], [-0.0014017501435378563 + 0.015997651719406848*1j, -0.005029118156235806 + 0.008769398580301738*1j, 0.013744020898215255 + 0.007270066767596865*1j, -0.00013027817160183783 + 0.03263360521160989*1j, -0.00656050349481813 + 0.01459867775444012*1j, 0. + 0.*1j, 0.006812086514330952 + 0.005998865776755266*1j, -0.006812086514330945 - 0.0059988657767552625*1j, -0.00028386385567872804 - 0.024794929747943593*1j, 0.0014722737077573994 + 0.00688923178513011*1j, 0.002004059812261427 + 0.00835382369542702*1j, 0.9393540936997326 + 0.3384776215043008*1j, -0.0050349541305219625 - 0.003188820652252622*1j, -0.0014722737077574027 - 0.0068892317851300955*1j, 0.0010441801531527223 - 0.0034979785475647194*1j, -0.006505510007840765 + 0.008940766615165916*1j], [-0.0023847390762034727 - 0.007474120643855278*1j, 0.5438046323966192 + 0.37250877020823536*1j, 0.19167107208905637 + 0.47889766775016607*1j, 0.004286152226219195 + 0.06736278925890729*1j, 0.14618340557564802 - 0.2257861406737896*1j, 0. + 0.*1j, 0.008787246909836913 + 0.007243548836888165*1j, -0.008787246909836894 - 0.007243548836888163*1j, 0.19159928291626968 - 0.08224803307094462*1j, -0.007178724114563575 + 0.008022024485799346*1j, -0.005112895313638863 + 0.011197038862113467*1j, -0.0025772448515652686 - 0.005027136169963416*1j, 0.41373844537289695 + 0.07510367916804545*1j, 0.007178724114563587 - 0.00802202448579939*1j, 0.0008245494782927158 + 0.02365898528626828*1j, 0.006865590686911381 - 0.02982888797841833*1j], [-0.01385424060946873 + 0.007346766056902622*1j, 0.007253387712084646 - 0.026398231595981106*1j, -0.005042397282404355 + 0.0006799644704053858*1j, 0.0063554354361674455 + 0.0024519591745773566*1j, 0.010084659341088932 + 0.0012477256489935814*1j, 0. + 0.*1j, -0.0010674253644799747 + 0.0038681181003789892*1j, 0.0010674253644799771 - 0.003868118100378991*1j, -0.006925061112831578 - 0.007209360953522748*1j, 0.0004445333398057167 + 0.0012634593200081508*1j, -0.00556518910749318 + 0.004171733577648479*1j, 0.005481323005962598 - 0.004302387118676577*1j, 0.002496558257397097 + 0.01217440859457674*1j, 0.9354522903175008 + 0.3510107739473203*1j, 0.00909614583538204 - 0.000272572191161386*1j, -0.009012279733851459 + 0.00040322573218949084*1j], [-0.0014017501435378602 + 0.01599765171940686*1j, -0.006560503494818149 + 0.014598677754440183*1j, -0.0002838638556786864 - 0.024794929747943582*1j, -0.005034954130522047 - 0.003188820652252608*1j, -0.005029118156235765 + 0.008769398580301728*1j, 0. + 0.*1j, 0.0014722737077573985 + 0.006889231785130107*1j, -0.0014722737077574003 - 0.006889231785130111*1j, 0.013744020898215274 + 0.007270066767596878*1j, 0.006812086514330947 + 0.0059988657767552685*1j, 0.0020040598122614316 + 0.008353823695427013*1j, 0.00104418015315272 - 0.0034979785475647077*1j, -0.0001302781716018669 + 0.03263360521160993*1j, -0.006812086514330952 - 0.005998865776755259*1j, 0.9393540936997327 + 0.33847762150430055*1j, -0.006505510007840765 + 0.008940766615165924*1j], [-0.00589729098294034 - 0.02078795330013387*1j, 0.0032128332326649502 - 0.015169745812674091*1j, 0.009648173932384308 - 0.008657321983380217*1j, 0.016590389853615075 - 0.02422696843972474*1j, 0.003212833232664965 - 0.01516974581267402*1j, 0. + 0.*1j, -0.0066568340250699395 - 0.005999597451123382*1j, 0.006656834025069941 + 0.005999597451123376*1j, 0.009648173932384391 - 0.008657321983380194*1j, -0.006656834025069943 - 0.005999597451123378*1j, -0.003203690427937027 - 0.013505261160819383*1j, 0.0008195139153344008 + 0.01159044858082166*1j, 0.016590389853615082 - 0.024226968439724755*1j, 0.006656834025069945 + 0.0059995974511233684*1j, 0.0008195139153343936 + 0.011590448580821668*1j, 0.9374614862545747 + 0.342598597266505*1j]])

        controls = np.array([0.1, 0.1, 0.1, 0.1])
        system = rydberg_atoms(2, self.rvec2, self.mu, self.delta, self.omega)
        self.assertArrayEqual(target, system.u(0.9, controls))

    def test_nz_does_not_blow_up_one(self):
        # a case that was empirically found to cause problems previously,
        # preserved as test case to catch regressions

        r = 4.0
        r0 = np.array([0.72480473, 0.08708385, -0.87074443])
        rvec = r*r0/np.linalg.norm(r0)

        a = rydberg_atoms(3, rvec, 2.0, -200.0, 1.0, nz=219, max_nz=221)

        ctrl = np.array([0.00840315, 0.02027272, -0.03879325, 0.01801073, 0.03176829, -0.02522575])

        try:
            a.u(1.0, ctrl)
        except floq.NZTooLargeError:
            self.fail("nz blew up unexpectedly!")


    def test_nz_does_not_blow_up_two(self):
        # a case that was empirically found to cause problems previously,
        # preserved as test case to catch regressions
        r = 5.0
        r0 = np.array([0.72480473, 0.08708385, -0.87074443])
        rvec = r*r0/np.linalg.norm(r0)
        a = rydberg_atoms(3, rvec, 2.0, -200.0, omega=1.0, nz=207, max_nz=211)
        ctrl = np.array([-0.01672054, 0.01984895, -0.05714435,
                         0.02228267, -0.00592705, -0.00610819])
        try:
            a.u(1.0, ctrl)
        except floq.NZTooLargeError:
            self.fail("nz blew up unexpectedly!")
