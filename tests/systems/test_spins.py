from tests.assertions import CustomAssertions
import numpy as np
import floq
import importlib.machinery
import importlib.util

# Import the spins example file as a module, even though it's outside the
# standard package structure.
loader = importlib.machinery.SourceFileLoader('spins', 'examples/spins.py')
spec = importlib.util.spec_from_loader(loader.name, loader)
spins = importlib.util.module_from_spec(spec)
loader.exec_module(spins)

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
        result = spins.hamiltonian(2, freq, controls)
        self.assertArrayEqual(target, result)

class TestSpindHf(CustomAssertions):
    def test_build_single_dhf(self):
        amp = 1.25
        target = dhf()
        result = spins.dhamiltonian(2)
        self.assertArrayEqual(target, result)

class TestSpinEnsemble(CustomAssertions):
    def setUp(self):
        self.amps = np.array([1.2, 1.1, 0.7, 0.6])
        self.freqs = np.array([0.8, 1.1, 0.9, 1.2])
        self.ensemble = spins.SpinEnsemble(4, 2, 1.0, self.freqs, self.amps)
        self.controls = np.array([1.5, 1.3, 1.4, 1.1])
        self.t = 3.0

    def test_systems_works(self):
        self.assertIsInstance(self.ensemble.systems, list)

    def test_single_system_evolves_correctly(self):
        system = self.ensemble.systems[0]
        result = system.u(self.t, self.controls)
        single = spins.spin(2, self.amps[0], self.freqs[0], omega=1.0)
        target = single.u(self.t, self.controls)
        self.assertArrayEqual(result, target, decimals=10)
