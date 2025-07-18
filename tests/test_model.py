import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import model


def test_model_shape_and_forward():
    m = model.create_model()
    params = m.count_params()
    assert 15_200_000 < params < 15_400_000
    x = np.zeros((1, 6170), dtype=np.float32)
    y = m(x)
    assert y.shape == (1, 100)