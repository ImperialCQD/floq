from unittest import TestCase
from tests.assertions import CustomAssertions
import floq.optimization.fidelity as fid
from floq.systems import SpinEnsemble
import numpy as np
from mock import MagicMock


class TestFidelityBaseIterations(TestCase):
    def setUp(self):
        self.computer = fid.FidelityBase(None)
        self.computer._f = MagicMock()
        self.computer._df = MagicMock()

    def test_increase_iterations(self):
        self.computer.iterate(None)
        self.computer.iterate(None)
        self.computer.iterate(None)
        self.computer.iterate(None)
        self.assertEqual(self.computer.iterations, 4)

    def test_reset(self):
        self.computer.reset_iterations()
        self.assertEqual(self.computer.iterations, 0)

class TestEnsembleFidelity(CustomAssertions):
    def setUp(self):
        self.ensemble = SpinEnsemble(2, 2, 1.5, np.array([1.1, 1.1]), np.array([1, 1]))

    def test_correct_in_one_case(self):
        target = np.array([[0.105818 - 0.324164j, -0.601164 - 0.722718j],
                           [0.601164 - 0.722718j, 0.105818 + 0.324164j]])

        f = fid.EnsembleFidelity(self.ensemble, fid.OperatorDistance, t=1.0, target=target)
        print(f.f(np.array([1.5, 1.5, 1.5, 1.5])))
        self.assertTrue(np.isclose(0.0, f.f(np.array([1.5, 1.5, 1.5, 1.5])), atol=1e-5))
