import sys
import numpy as np
import types

# Provide a dummy 'flwr' module so importing experiment_runner does not fail
sys.modules.setdefault("flwr", types.ModuleType("flwr"))

from src import membership_attack
from src.experiment_runner import membership_attack_scores
from src.model import create_model


def test_membership_attack_uses_provided_model(monkeypatch):
    seen = []

    def fake_compute_gradient_sum(model):
        """Fake implementation of compute_gradient_sum for unit tests
        
        This implementation records the passed model in the list 'seen' and
        returns zero gradients with the same shapes as the model's weights.
        """
        seen.append(model)
        # return zeros with correct shapes
        return [np.zeros_like(w.numpy()) for w in model.trainable_variables]

    monkeypatch.setattr(membership_attack, "compute_gradient_sum", fake_compute_gradient_sum)
    # ensure experiment_runner uses the patched version as well
    from src import experiment_runner
    monkeypatch.setattr(experiment_runner, "compute_gradient_sum", fake_compute_gradient_sum)

    model = create_model()
    data_x = np.zeros((1, 6170), dtype=np.float32)
    data_y = np.zeros((1,), dtype=np.int64)
    update = [np.zeros_like(w.numpy()) for w in model.trainable_variables]

    membership_attack_scores(model, update, (data_x, data_y), (data_x, data_y))

    assert len(seen) == 2  # called once for pos and once for neg
    assert all(m is model for m in seen)