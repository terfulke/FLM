import json
import flwr as fl
from dataProcessing import load_and_preprocess, make_client_splits
from client import get_client_fn
from strategy import HybridSyncAsyncStrategy

def aggregate_accuracy(metrics):
    accuracies = [m.get("accuracy", 0) for _, m in metrics]
    if not accuracies:
        return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / len(accuracies)}

if __name__ == "__main__":
    NUM_CLIENTS = 5
    ASYNC_CLIENT_IDS = ["2", "4"]  # Example async clients
    LOCAL_EPOCHS = 1

    df = load_and_preprocess("credit_score_classification/train.csv")  # Your csv path here
    splits = make_client_splits(df, NUM_CLIENTS, strategy="iid")

    client_fn = get_client_fn(splits, local_epochs=LOCAL_EPOCHS, async_client_ids=ASYNC_CLIENT_IDS)

    strategy = HybridSyncAsyncStrategy(
        async_client_ids=ASYNC_CLIENT_IDS,
        async_weight=0.3,
        evaluate_metrics_aggregation_fn=aggregate_accuracy,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

    # Print full metrics dictionary (accuracy per round)
    print(history.metrics_distributed)

    # Save metrics dictionary to JSON file
    with open("hybrid_experiment_results.json", "w") as f:
        json.dump(history.metrics_distributed, f, indent=2)
