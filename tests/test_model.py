import numpy as np
from src import model


def test_model_shape_and_forward():
    """Test that the model has the expected number of parameters and
    that the forward pass produces a tensor of the correct shape.
    """
    m = model.create_model()
    # The model should have roughly 15.3 million parameters
    params = m.count_params()
    assert 15_200_000 < params < 15_400_000
    # Create a dummy input tensor
    x = np.zeros((1, 6170), dtype=np.float32)
    # Run the model on the dummy input
    y = m(x)
    # The output should have shape (batch_size, 100)
    assert y.shape == (1, 100)
