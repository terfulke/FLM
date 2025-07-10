import json
import flwr as fl
from strategy import FedSAStrategy
from dataProcessing import load_and_preprocess, make_client_splits
from client import get_client_fn

# Accuracy aggregation function (weighted average)
def aggregate_accuracy(metrics):
    total, correct = 0, 0
    for num_examples, m in metrics:
        if "accuracy" in m:
            correct += m["accuracy"] * num_examples
            total += num_examples
    return {"accuracy": correct / total if total > 0 else 0.0}

# Define experiment configurations
tests = [
    {"name": "5_clients_iid_E1_async",   "num_clients": 5,  "split": "iid",    "local_epochs": 1, "async_weight": 0.3},
    {"name": "10_clients_iid_E1_async",  "num_clients": 10, "split": "iid",    "local_epochs": 1, "async_weight": 0.3},
    {"name": "5_clients_noniid_E1_async","num_clients": 5,  "split": "noniid", "local_epochs": 1, "async_weight": 0.3},
    {"name": "5_clients_iid_E5_async",   "num_clients": 5,  "split": "iid",    "local_epochs": 5, "async_weight": 0.3},
]

# Load data once
df = load_and_preprocess("credit_score_classification/train.csv")
all_results = {}

for cfg in tests:
    name = cfg["name"]
    num_clients = cfg["num_clients"]

    print(f"\nRunning experiment: {name}")
    splits = make_client_splits(df, num_clients, strategy=cfg["split"])
    client_fn = get_client_fn(splits, local_epochs=cfg["local_epochs"])

    strategy = FedSAStrategy(
        M=3,
        tau_0=2,
        base_lr=1.0,
        num_total_clients=num_clients,
        evaluate_metrics_aggregation_fn=aggregate_accuracy,
        fraction_fit=0.3,
        min_fit_clients=1,
        min_available_clients=num_clients,
    )


    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # Save losses and accuracy
    all_results[name] = {
        "loss": history.losses_distributed,
        "accuracy": history.metrics_distributed,
    }

# Save all experiments to a file
with open("hybrid_experiment_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nAll hybrid experiments completed. Results saved to hybrid_experiment_results.json")
