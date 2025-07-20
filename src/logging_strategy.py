import os
from typing import List, Tuple

import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that logs per-client updates each round."""

    def __init__(self, log_dir: str = "logs/updates", **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._current_weights: List[np.ndarray] | None = None

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        params = super().initialize_parameters(client_manager)
        if params is not None:
            self._current_weights = parameters_to_ndarrays(params)
        return params

    def aggregate_fit(
        self,
        server_round: int,
        results: list[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]] | List[BaseException],
    ) -> Tuple[fl.common.Parameters | None, dict]:
        # Log each client's update before aggregation
        if self._current_weights is not None:
            for proxy, fit_res in results:
                cid = fit_res.metrics.get("cid", "unknown")
                weights_nd = parameters_to_ndarrays(fit_res.parameters)
                update = [nw - ow for nw, ow in zip(weights_nd, self._current_weights)]
                fname = f"round{server_round:03d}_client{cid}.npz"
                path = os.path.join(self.log_dir, fname)
                np.savez_compressed(path, *update)

        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            self._current_weights = parameters_to_ndarrays(aggregated_params)
        return aggregated_params, aggregated_metrics