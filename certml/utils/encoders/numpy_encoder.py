"""Numpy Encoder"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Numpy Encoder

    Encode numpy object as JSON.
    """

    def default(self, obj):
        """Override default encoder

        If the object is a 1D numpy array, encode it as JSON array.
        If the object is a numpy float, encode it as a JSON number.

        Parameters
        ----------
        obj : nd.array, np.floating, or other type
            Object to encode

        Returns
        -------
        json : str
            Encoded JSON blob
        """
        if isinstance(obj, np.ndarray):
            assert len(np.shape(obj)) == 1  # Can only handle 1D ndarrays
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(NumpyEncoder, self).default(obj)
