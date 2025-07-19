import argparse
import os
import numpy as np
import flwr as fl
from logging_strategy import LoggingFedAvg
from model import create_model
from fl_client import TexasClient


def evaluate_fn_factory(x_test: np.ndarray, y_test: np.ndarray):
    def evaluate(server_round: int, parameters, config):
        model = create_model()
        model.set_weights(parameters)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Server evaluation round {server_round}: loss={loss:.4f} acc={acc:.4f}")
        if server_round == 100:
            os.makedirs("logs", exist_ok=True)
            model.save(os.path.join("logs", "final_model.h5"))
        return loss, {"accuracy": float(acc)}

    return evaluate


def fit_config(server_round: int):
    return {"server_round": server_round}


def start_server(address: str, log_dir: str = "logs/client_updates"):
    data = np.load(os.path.join("data", "texas100_subset.npz"))
    evaluate_fn = evaluate_fn_factory(data["X_test"], data["y_test"])
    strategy = LoggingFedAvg(
        log_dir = log_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_fn,
    )
    fl.server.start_server(server_address=address, config={"num_rounds": 100}, strategy=strategy)


def start_client(address: str, cid: int, data_dir: str):
    client = TexasClient(cid, data_dir=data_dir)
    fl.client.start_numpy_client(server_address=address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated training runner")
    parser.add_argument("--role", choices=["server", "client"], required=True)
    parser.add_argument("--cid", type=int, default=0, help="Client id")
    parser.add_argument("--address", default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--data-dir", default="data/clients", help="Client data directory")
    parser.add_argument("--log-dir", default="logs/client_updates", help="Directory to store per-client updates")
    args = parser.parse_args()

    if args.role == "server":
        start_server(args.address, log_dir=args.log_dir)
    else:
        start_client(args.address, args.cid, args.data_dir)