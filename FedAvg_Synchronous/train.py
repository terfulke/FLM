import flwr as fl
from client import client_fn

if __name__ == "__main__":
    NUM_CLIENTS = 5
    # Run FedAvg for 5 rounds synchronously
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
    )
    print(history.metrics_aggregated)
    print(history.loss_distributed)
