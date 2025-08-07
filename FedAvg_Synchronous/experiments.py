import json
import flwr as fl
from flwr.server.strategy import FedAvg

# Use the correct module names matching your files
from dataProcessing import load_and_preprocess, make_client_splits
from client import get_client_fn

# Function to aggregate accuracy metrics across clients
def aggregate_accuracy(metrics):
    """
    Compute the average accuracy from per-client metrics.
    Args:
        metrics: List of tuples (client_id, metrics_dict)
    Returns:
        Dict with aggregated accuracy.
    """
    accuracies = [m.get("accuracy", 0) for _, m in metrics]
    if not accuracies:
        return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / len(accuracies)}

# Define the federated learning strategy with accuracy aggregation
strategy = FedAvg(
    evaluate_metrics_aggregation_fn=aggregate_accuracy
)

# Experiment configurations
tests = [
    {"name": "5_clients_iid_E1",   "num_clients": 5,  "split": "iid",    "local_epochs": 1},
    {"name": "10_clients_iid_E1",  "num_clients": 10, "split": "iid",    "local_epochs": 1},
    {"name": "5_clients_noniid_E1","num_clients": 5,  "split": "noniid", "local_epochs": 1},
    {"name": "5_clients_iid_E5",   "num_clients": 5,  "split": "iid",    "local_epochs": 5},
]

# Load & preprocess once
df = load_and_preprocess("/Users/sagarsikdar/projects/FLM/dataSets/credit-score-classification/train.csv")
all_results = {}

for cfg in tests:
    name = cfg["name"]
    n_clients = cfg["num_clients"]
    # Split data according to strategy
    splits = make_client_splits(df, n_clients, strategy=cfg["split"])
    # Build a client function using the factory from client.py
    client_fn = get_client_fn(splits, local_epochs=cfg["local_epochs"])

    print(f"\nRunning experiment: {name}")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # Store metrics
    all_results[name] = {
        "loss": history.losses_distributed,
        "accuracy": history.metrics_distributed,
    }

# Save experiment results
with open("experiment_results.json", "w") as fp:
    json.dump(all_results, fp, indent=2)

print("\nAll experiments completed. Results saved to experiment_results.json")
