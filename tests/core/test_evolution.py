from tests.assertions import CustomAssertions
import numpy as np
import tests.rabi as rabi
import floq.core.evolution as ev
import floq.helpers.index as h
import floq.core.fixed_system as fs
import floq.errors as er

def generate_fake_spectrum(unique_vals, dim, omega, nz):
    vals = np.array([])
    for i in range(nz):
        offset = h.i_to_n(i, nz)
        new = unique_vals + offset * omega * np.ones(dim)
        vals = np.append(vals, new)
    return vals

class TestAssembleK(CustomAssertions):
    def setUp(self):
        dim = 2
        self.p = fs.FixedSystemParameters(dim, nz=5, nc=3, omega=1)
        a = -1.*np.ones([dim, dim])
        b = np.zeros([dim, dim])
        c = np.ones([dim, dim])
        z = np.zeros([dim, dim])
        i = np.identity(dim)

        self.goalk = np.array(
            np.bmat(
                [[b-2*i, a, z, z, z],
                 [c, b-i, a, z, z],
                 [z, c, b, a, z],
                 [z, z, c, b+i, a],
                 [z, z, z, c, b+2*i]]))
        self.hf = np.array([a, b, c])

    def test_build(self):
        builtk = ev.assemble_k(self.hf, self.p)
        self.assertArrayEqual(builtk, self.goalk)

class TestAssembledK(CustomAssertions):
    def setUp(self):
        dim = 2
        self.p = fs.FixedSystemParameters(dim, 5, 3, np=2, omega=1)

        a = -1.*np.ones([dim, dim])
        b = np.zeros([dim, dim])
        c = np.ones([dim, dim])
        z = np.zeros([dim, dim])
        i = np.identity(dim)

        dk1 = np.array(
            np.bmat(
                [[b, a, z, z, z],
                 [c, b, a, z, z],
                 [z, c, b, a, z],
                 [z, z, c, b, a],
                 [z, z, z, c, b]]))
        dk2 = np.array(
            np.bmat(
                [[b, b, z, z, z],
                 [a, b, b, z, z],
                 [z, a, b, b, z],
                 [z, z, a, b, b],
                 [z, z, z, a, b]]))

        self.goaldk = np.array([dk1, dk2])
        self.dhf = np.array([[a, b, c], [b, b, a]])

    def test_build(self):
        builtdk = ev.assemble_dk(self.dhf, self.p)
        self.assertArrayEqual(builtdk, self.goaldk)

class TestFindEigensystem(CustomAssertions):
    def setUp(self):
        self.target_vals = np.array([-0.235, 0.753])
        # random matrix with known eigenvalues:
        # {-1.735, -0.747, -0.235, 0.753, 1.265, 2.253}
        k = np.array([[-0.0846814, -0.0015136 - 0.33735j, -0.210771 + 0.372223j,
                       0.488512 - 0.769537j, -0.406266 + 0.315634j, -0.334452 +
                        0.251584j], [-0.0015136 + 0.33735j,
                       0.809781, -0.416533 - 0.432041j, -0.571074 -
                        0.669052j, -0.665971 + 0.387569j, -0.297409 -
                        0.0028969j], [-0.210771 - 0.372223j, -0.416533 +
                        0.432041j, -0.0085791, 0.110085 + 0.255156j,
                       0.958938 - 0.17233j, -0.91924 + 0.126004j], [0.488512 +
                        0.769537j, -0.571074 + 0.669052j,
                       0.110085 - 0.255156j, -0.371663,
                       0.279778 + 0.477653j, -0.496302 + 1.04898j], [-0.406266 -
                        0.315634j, -0.665971 - 0.387569j, 0.958938 + 0.17233j,
                       0.279778 - 0.477653j, -0.731623,
                       0.525248 + 0.0443422j], [-0.334452 - 0.251584j, -0.297409 +
                        0.0028969j, -0.91924 - 0.126004j, -0.496302 - 1.04898j,
                       0.525248 - 0.0443422j, 1.94077]], dtype='complex128')

        e1 = np.array([[0.0321771 - 0.52299j, 0.336377 + 0.258732j],
                       [0.371002 + 0.0071587j, 0.237385 + 0.205185j],
                       [0.525321 + 0.j, 0.0964822 + 0.154715j]])
        e2 = np.array([[0.593829 + 0.j, -0.105998 - 0.394563j],
                       [-0.0737891 - 0.419478j, 0.323414 + 0.350387j],
                       [-0.05506 - 0.169033j, -0.0165495 + 0.199498j]])
        self.target_vecs = np.array([e1, e2])

        omega = 2.1
        nz = 3
        dim = 2
        p = fs.FixedSystemParameters(dim, nz, omega=omega, decimals=3, sparse=False)
        self.vals, self.vecs = ev.find_eigensystem(k, p)

    def test_finds_vals(self):
        self.assertArrayEqual(self.vals, self.target_vals)

    def test_finds_vecs(self):
        self.assertArrayEqual(self.vecs, self.target_vecs, decimals=3)

    def test_casts_as_complex128(self):
        self.assertEqual(self.vecs.dtype, 'complex128')

class TestFindDuplicates(CustomAssertions):
    def test_duplicates(self):
        a = np.array([1, 2.001, 2.003, 1.999, 3])
        res = tuple(ev.find_duplicates(a, 2))
        self.assertEqual(len(res), 1)
        self.assertArrayEqual([1, 2, 3], res[0])

    def test_empty_if_no_dup(self):
        a = np.array([1, 2.001, 4.003, 8.999, 10])
        res = tuple(ev.find_duplicates(a, 2))
        self.assertEqual(res, ())

class TestCalculatePhi(CustomAssertions):
    def test_sum(self):
        a = np.array([1.53, 2.45])
        b = np.array([7.161, 1.656])
        c = np.array([2.3663, 8.112])
        e1 = np.array([a, a, c])
        e1_sum = a+a+c
        e2 = np.array([c, a, b])
        e2_sum = c+a+b
        target = np.array([e1_sum, e2_sum])
        calculated_sum = ev.calculate_phi(np.array([e1, e2]))
        self.assertArrayEqual(calculated_sum, target)

class TestCalculatePsi(CustomAssertions):
    def test_sum(self):
        omega = 2.34
        t = 1.22
        p = fs.FixedSystemParameters(2, 3, omega=omega, t=t)
        a = np.array([1.53, 2.45], dtype='complex128')
        b = np.array([7.161, 1.656], dtype='complex128')
        c = np.array([2.3663, 8.112], dtype='complex128')
        e1 = np.array([a, a, c])
        e1_sum = np.exp(-1j*omega*t)*a+a+np.exp(1j*omega*t)*c
        e2 = np.array([c, a, b])
        e2_sum = np.exp(-1j*omega*t)*c+a+np.exp(1j*omega*t)*b

        target = np.array([e1_sum, e2_sum])
        vecs = np.array([e1, e2])

        calculated_sum = ev.calculate_psi(vecs, p)

        self.assertArrayEqual(calculated_sum, target)


class TestCalculateU(CustomAssertions):
    def test_u(self):
        omega = 3.56
        t = 8.123
        dim = 2
        nz = 3
        p = fs.FixedSystemParameters(dim, nz, omega=omega, t=t)

        energies = [0.23, 0.42]

        e1 = np.array([1.563 + 1.893j, 1.83 + 1.142j, 0.552 + 0.997j,
                       0.766 + 1.162j, 1.756 + 0.372j, 0.689 + 0.902j])
        e2 = np.array([1.328 + 1.94j, 1.866 + 0.055j, 1.133 + 0.162j,
                       1.869 + 1.342j, 1.926 + 1.587j, 1.735 + 0.942j])
        vecs = np.array([e1, e2])
        vecs = np.array([np.split(eva, nz) for eva in vecs])

        phi = ev.calculate_phi(vecs)
        psi = ev.calculate_psi(vecs, p)

        target = np.array([[29.992 + 14.079j, 29.125 + 18.169j],
                           [5.117 - 1.363j, 5.992 - 2.462j]]).round(3)
        u = ev.calculate_u(phi, psi, energies, p).round(3)

        self.assertArrayEqual(u, target)


class TestCalculateUdot(CustomAssertions):
    def test_udot(self):
        omega = 2.56
        t = 1.23
        dim = 2
        nz = 3
        p = fs.FixedSystemParameters(dim, nz, omega=omega, t=t)

        energies = [0.23, 0.42]

        e1 = np.array([1.563 + 1.893j, 1.83 + 1.142j, 0.552 + 0.997j,
                       0.766 + 1.162j, 1.756 + 0.372j, 0.689 + 0.902j])
        e2 = np.array([1.328 + 1.94j, 1.866 + 0.055j, 1.133 + 0.162j,
                       1.869 + 1.342j, 1.926 + 1.587j, 1.735 + 0.942j])
        vecs = np.array([e1, e2])
        vecs = np.array([np.split(eva, nz) for eva in vecs])

        phi = ev.calculate_phi(vecs)
        psi = ev.calculate_psi(vecs, p)
        psidot = ev.calculate_psidot(vecs, p)

        target = np.array([[-18.3738 + 28.8951j, -19.4244 + 25.5303j], [22.7487 + 1.48245j, 25.7424 + 2.79668j]])
        target = target.round(3)
        u = ev.calculate_udot(phi, psi, psidot, energies, p).round(3)

        self.assertArrayEqual(u, target)
