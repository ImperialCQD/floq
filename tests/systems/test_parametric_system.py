from unittest import TestCase
import numpy as np
import floq
from tests.assertions import CustomAssertions
from mock import MagicMock, patch

class TestParametricSystemBaseCaching(TestCase):
    def setUp(self):
        self.ctrls1 = np.array([1.2, 1.1])
        self.ctrls2 = np.array([1.3, 1.1])

        self.real = floq.System(MagicMock(), MagicMock(), nz=MagicMock(),
                omega=MagicMock())
        self.real._System__fixed = MagicMock()

    def test_u_caches_if_same(self):
        with patch('floq.core.FixedSystem') as mock:
            self.real.u(self.ctrls1, 1.0)
            self.real.u(self.ctrls1, 1.0)
            mock.assert_called_once()

    def test_u_does_not_cache_if_not_same(self):
        with patch('floq.core.FixedSystem') as mock:
            self.real.u(self.ctrls1, 1.0)
            self.real.u(self.ctrls2, 1.0)
            self.assertEqual(mock.call_count, 2)
