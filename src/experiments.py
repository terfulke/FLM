import json, random
import torch
from dataProcessing import load_and_preprocess, make_client_splits
from client import get_client_fn

# ---------------- data -----------------
df = load_and_preprocess("datasets/train.csv")   # <-- adjust path if needed

# ------------- FedBuff core ------------
def run_fedbuff_experiment(num_clients, split, local_epochs, buffer_size, max_rounds=5):
    # build clients
    splits     = make_client_splits(df, num_clients, strategy=split)
    client_fn  = get_client_fn(splits, local_epochs=local_epochs)
    clients    = [client_fn(str(cid)) for cid in range(num_clients)]

    # start from one client's weights
    global_weights = clients[0].get_parameters()

    loss_hist, acc_hist = [], []
    rounds_done         = 0
    order               = list(range(num_clients))
    random.shuffle(order)
    pos = 0

    while rounds_done < max_rounds:
        buf_params, buf_samples = [], []

        # --- collect updates until buffer full ---
        while len(buf_params) < buffer_size:
            if pos >= len(order):                     # reshuffle when finished a pass
                random.shuffle(order); pos = 0
            cid = order[pos];  pos += 1
            p, n, _ = clients[cid].fit(global_weights, config={})  # <-- unpack 3 values
            buf_params.append(p)
            buf_samples.append(n)
            if len(buf_params) == buffer_size: break

        # --- aggregate FedBuff style (weighted average) ---
        total = sum(buf_samples)
        new_weights = []
        for idx in range(len(buf_params[0])):                 # iterate over each tensor
            w_sum = sum(buf_params[j][idx] * buf_samples[j]   # weighted sum
                        for j in range(len(buf_params)))
            new_weights.append(w_sum / total)
        global_weights = new_weights
        rounds_done += 1

        # --- evaluate on every client’s validation split ---
        tot_loss, tot_samples, tot_acc = 0.0, 0, 0.0
        for cl in clients:
            v_loss, v_cnt, metr = cl.evaluate(global_weights, config={})
            if v_cnt == 0: continue
            tot_loss   += v_loss * v_cnt
            tot_samples += v_cnt
            tot_acc    += metr.get("accuracy", 0.0)
        loss_hist.append([rounds_done, tot_loss / tot_samples])
        acc_hist .append([rounds_done, tot_acc / num_clients])   # mean of accuracies

    return {"loss": loss_hist, "accuracy": {"accuracy": acc_hist}}

# -------------- experiments -------------
experiments = [
    {"name": "5_clients_iid_E1",      "n": 5,  "split": "iid",    "E": 1, "B": 3},
    {"name": "10_clients_iid_E1",     "n":10,  "split": "iid",    "E": 1, "B": 5},
    {"name": "5_clients_noniid_E1",   "n": 5,  "split": "noniid", "E": 1, "B": 3},
    {"name": "10_clients_noniid_E1",  "n":10,  "split": "noniid", "E": 1, "B": 5},
    {"name": "5_clients_iid_E5",      "n": 5,  "split": "iid",    "E": 5, "B": 3},
]

results = {}
for cfg in experiments:
    print(f"Running FedBuff experiment: {cfg['name']}")
    results[cfg["name"]] = run_fedbuff_experiment(
        num_clients  = cfg["n"],
        split        = cfg["split"],
        local_epochs = cfg["E"],
        buffer_size  = cfg["B"],
    )

with open("fedbuff_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ FedBuff experiments complete. Saved to fedbuff_results.json")
