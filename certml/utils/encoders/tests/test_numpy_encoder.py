"""Numpy Encoder Unit Tests"""

import numpy as np
from certml.utils.encoders import NumpyEncoder


class TestNumpyEncoder(object):
    """Numpy Encoder Unit Tests"""

    def test_np_1d_array(self):
        blob = NumpyEncoder().encode(np.array([1, 2, 3]))
        assert blob == '[1, 2, 3]'

    def test_np_float(self):
        blob = NumpyEncoder().encode(np.float_(1.0))
        assert blob == '1.0'

    def test_non_np(self):
        blob = NumpyEncoder().encode([1, 2, 3])
        assert blob == '[1, 2, 3]'
