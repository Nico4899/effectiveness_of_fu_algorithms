import argparse
import os

import flwr as fl

from fl_client import TexasClient


def start_server() -> None:
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=10,
        min_eval_clients=10,
        min_available_clients=10,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
    )


def start_client(client_id: int) -> None:
    client = TexasClient(client_id)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--client-id", type=int, default=None)
    args = parser.parse_args()

    if args.role == "server":
        start_server()
    else:
        cid = args.client_id
        if cid is None:
            proc_id = int(os.environ.get("SLURM_PROCID", "1"))
            cid = proc_id - 1
        start_client(cid)


if __name__ == "__main__":
    main()