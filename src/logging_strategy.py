# Custom Flower strategy to record per-client updates before aggregation

from __future__ import annotations

import os
from typing import List, Tuple, Union, Dict

import numpy as np
import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """Federated averaging strategy that logs client updates."""

    def __init__(self, log_dir: str = "logs/client_updates", **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Parameters | None, Dict[str, fl.common.Scalar]]:
        """Aggregate fit results and record each client's update."""

        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Current global parameters as ndarrays
        current_params = None
        if self.initial_parameters is not None and server_round == 1:
            current_params = parameters_to_ndarrays(self.initial_parameters)
        elif hasattr(self, "_last_aggregated") and self._last_aggregated is not None:
            current_params = parameters_to_ndarrays(self._last_aggregated)

        for proxy, fit_res in results:
            client_id = proxy.cid
            updates = parameters_to_ndarrays(fit_res.parameters)
            if current_params is not None:
                delta = [u - c for u, c in zip(updates, current_params)]
            else:
                # If we do not know the previous global params, store raw update
                delta = updates
            path = os.path.join(self.log_dir, f"round{server_round}_client{client_id}.npz")
            np.savez(path, *delta)

        aggregated_ndarrays = (
            fl.server.strategy.fedavg.aggregate_inplace(results)
            if self.inplace
            else fl.server.strategy.fedavg.aggregate(
                [
                    (parameters_to_ndarrays(res.parameters), res.num_examples)
                    for _, res in results
                ]
            )
        )

        aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self._last_aggregated = aggregated

        metrics_aggregated: Dict[str, fl.common.Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return aggregated, metrics_aggregated