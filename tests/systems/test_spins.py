from tests.assertions import CustomAssertions
import numpy as np
import floq

def single_hf(controls, omega):
    a1 = controls[0]
    b1 = controls[1]
    a2 = controls[2]
    b2 = controls[3]
    return np.array([[[0, 0.25*(1j*a2 + b2)],
                      [0.25*1j*(a2 + 1j*b2), 0]],
                     [[0, 0.25*(1j*a1 + b1)],
                      [0.25*1j*(a1 + 1j*b1), 0]],
                     [[omega/2.0, 0],
                      [0, -(omega/2.0)]],
                     [[0, -0.25j*(a1 - 1j*b1)],
                      [0.25*(-1j*a1 + b1), 0]],
                     [[0, -0.25j*(a2 - 1j*b2)],
                     [0.25*(-1j*a2 + b2), 0]]])

def dhf():
    dhf_b1 = np.array([[[0., 0.], [0., 0.]],
                       [[0., 0.25], [-0.25, 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., -0.25], [0.25, 0.]],
                       [[0., 0.], [0., 0.]]])

    dhf_a1 = np.array([[[0., 0.], [0., 0.]],
                       [[0., 0. + 0.25j], [0. + 0.25j, 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0. - 0.25j], [0. - 0.25j, 0.]],
                       [[0., 0.], [0., 0.]]])

    dhf_b2 = np.array([[[0., 0.25], [-0.25, 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., -0.25], [0.25, 0.]]])

    dhf_a2 = np.array([[[0., 0. + 0.25j], [0. + 0.25j, 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0.], [0., 0.]],
                       [[0., 0. - 0.25j], [0. - 0.25j, 0.]]])
    return np.array([dhf_a1, dhf_b1, dhf_a2, dhf_b2])


class TestSpinHf(CustomAssertions):
    def test_build_single_hf(self):
        controls = np.array([1.2, 2.3, 3.4, 5.4])
        freq = 2.5
        target = single_hf(controls, freq)
        result = floq.systems._spin_hamiltonian(2, freq, controls)
        self.assertArrayEqual(target, result)

class TestSpindHf(CustomAssertions):
    def test_build_single_dhf(self):
        amp = 1.25
        target = dhf()
        result = floq.systems._spin_d_hamiltonian(2)
        self.assertArrayEqual(target, result)

class TestRandomisedSpinEnsemble(CustomAssertions):
    def test_init(self):
        # Not a very sophisticated test -- at least verify that it's not broken
        rand = floq.systems.RandomisedSpinEnsemble(10, 10, 1.0, 8.2, 0.25)
        self.assertIsInstance(rand.systems, list)

class TestSpinEnsemble(CustomAssertions):
    def setUp(self):
        self.amps = np.array([1.2, 1.1, 0.7, 0.6])
        self.freqs = np.array([0.8, 1.1, 0.9, 1.2])
        self.ensemble = floq.systems.SpinEnsemble(4, 2, 1.0, self.freqs,
                                                  self.amps)
        self.controls = np.array([1.5, 1.3, 1.4, 1.1])
        self.t = 3.0

    def test_systems_works(self):
        self.assertIsInstance(self.ensemble.systems, list)

    def test_single_system_evolves_correctly(self):
        system = self.ensemble.systems[0]
        result = system.u(self.controls, self.t)
        single = floq.systems.spin(2, self.amps[0], self.freqs[0], omega=1.0)
        target = single.u(self.controls, self.t)
        self.assertArrayEqual(result, target, decimals=10)
