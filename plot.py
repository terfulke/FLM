import json
import matplotlib.pyplot as plt

# Load results
with open("hybrid_experiment_results.json", "r") as f:
    results = json.load(f)

# Accuracy Plot
plt.figure(figsize=(5, 4))
for test, info in results.items():
    acc_data = info["accuracy"]["accuracy"]
    rounds = list(range(1, len(acc_data) + 1))
    accuracies = [acc for _, acc in acc_data]

    label = test.replace("_", " ")
    plt.plot(rounds, accuracies, marker="o", label=label)

plt.title("Accuracy over Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.ylim(0.4, 0.7)
plt.legend()
plt.savefig("best_accuracy_all_tests.png")
plt.show()

# Loss Plot
plt.figure(figsize=(5, 4))
for test, info in results.items():
    acc_data = info["loss"]
    rounds = list(range(1, len(acc_data) + 1))
    accuracies = [acc for _, acc in acc_data]

    label = test.replace("_", " ")
    plt.plot(rounds, accuracies, marker="o", label=label)

plt.title("Loss over Rounds")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.ylim(0.6, 1.0)
plt.legend()
plt.savefig("best_loss_all_tests.png")
plt.show()