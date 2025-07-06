# -------------------------------- fedbuff_async.py --------------------------------
"""
Pure-async FedBuff on the credit-score dataset (all in one file)

Usage:
  poetry run python fedbuff_async.py --buffer 5 --partition iid --rounds 60
"""
import argparse
import random
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Globals for saving
scaler: StandardScaler
feature_cols: List[str]

# ─────────────────────────── 1. Data loader & cleaning ─────────────────────────
def load_dataframe(
    path: str = "datasets/train.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global scaler, feature_cols
    t0 = time.time()
    log = lambda msg: print(f"[{datetime.now():%H:%M:%S}] {msg}")

    log("Reading CSV…")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    log(f"CSV loaded in {time.time()-t0:.1f}s")

    # Numeric coercion
    num_cols = [
        "Age", "Monthly_Inhand_Salary", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    log("Converting numeric columns…")
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Clean placeholders & drop invalid ages
    df.replace({"_": np.nan, "NA": np.nan}, inplace=True)
    df = df[df["Age"].gt(0, fill_value=False)]
    log(f"After Age filter → {len(df):,} rows")

    # Fill missing
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].mean())
    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    log("Missing values filled")

    # Drop high-cardinality & unwanted columns
    high_card = [c for c in df.select_dtypes(include="object").columns
                 if df[c].nunique() > 50]
    drop_cols = high_card + [
        "Credit_History_Age", "ID", "Customer_ID", "Month", "Name", "SSN"
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    log(f"Dropped {len(high_card)} high-cardinality cols")

    # Target encoding & one-hot encode categorical features
    y = df["Credit_Score"].astype("category").cat.codes.to_numpy()
    df.drop(columns=["Credit_Score"], inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    feature_cols = df.columns.tolist()
    log(f"One-hot dim = {df.shape}")

    # Scale features
    scaler = StandardScaler().fit(df.values.astype(np.float32))
    X = scaler.transform(df.values.astype(np.float32))
    log(f"Preprocessing done in {time.time()-t0:.1f}s")

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# ─────────────────────────── 2. Data splits ───────────────────────────────────
def split_iid(X, y, n=10):
    idx = np.random.permutation(len(X))
    return np.array_split(idx, n)

def split_label_skew(X, y, n=10):
    labs = np.unique(y)
    splits: List[np.ndarray] = []
    for cid in range(n):
        l = labs[cid % len(labs)]
        maj = np.where(y == l)[0][:600]
        mino = np.random.choice(np.where(y != l)[0], 200, replace=False)
        splits.append(np.concatenate([maj, mino]))
    return splits

# ─────────────────────────── 3. Model ─────────────────────────────────────────
class CreditNet(nn.Module):
    def __init__(self, d_in: int, n_cls: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_cls)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ─────────────────────────── 4. Client ────────────────────────────────────────
class Client(fl.client.NumPyClient):
    def __init__(
        self, cid: str, data: Tuple[np.ndarray, np.ndarray],
        d_in: int, n_cls: int, slow: bool = False
    ):
        self.cid = cid
        X, y = data
        # Store data as torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        # Initialize local model
        self.model = CreditNet(d_in, n_cls)
        self.slow = slow

    def _set(self, params: List[np.ndarray]) -> None:
        """Load parameters into the local model."""
        state = self.model.state_dict()
        for k, v in zip(state.keys(), params):
            state[k] = torch.tensor(v)
        self.model.load_state_dict(state, strict=True)

    def _get(self) -> List[np.ndarray]:
        """Get model parameters as a list of numpy arrays."""
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def get_parameters(self, config) -> fl.common.Parameters:
        return fl.common.ndarrays_to_parameters(self._get())

    def fit(self, parameters, config):
        # Set the model parameters for this round
        self._set(fl.common.parameters_to_ndarrays(parameters))
        # Simulate slower clients (every 3rd client is slow)
        if self.slow:
            time.sleep(random.uniform(1, 4))
        # Simple training loop (3 epochs)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for _ in range(3):
            opt.zero_grad()
            loss = F.cross_entropy(self.model(self.X), self.y)
            loss.backward()
            opt.step()
        # Return updated parameters and the number of samples
        return fl.common.ndarrays_to_parameters(self._get()), len(self.X), {}

    def evaluate(self, parameters, config):
        # Load global model parameters to evaluate
        self._set(fl.common.parameters_to_ndarrays(parameters))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X)
            loss = F.cross_entropy(logits, self.y).item()
            acc = (logits.argmax(1) == self.y).float().mean().item()
        # Return evaluation loss, number of samples, and a dict with accuracy
        return loss, len(self.X), {"acc": acc}

# ─────────────────────────── 5. FedBuff Strategy ─────────────────────────────
# ─────────────────────────── 5. FedBuff Strategy ─────────────────────────────
class FedBuff(fl.server.strategy.Strategy):
    """Pure asynchronous FedBuff (buffered FedAvg)."""

    def __init__(self, buffer_size: int, d_in: int, n_cls: int):
        super().__init__()
        self.B = buffer_size
        # create *correct-shape* initial parameters
        dummy = CreditNet(d_in, n_cls)
        self.params: fl.common.Parameters = fl.common.ndarrays_to_parameters(
            [v.cpu().numpy() for v in dummy.state_dict().values()]
        )
        self.buffer: List[Tuple[List[np.ndarray], int]] = []
        self.last_agg = time.time()

    # Flower ≥1.19 passes server_round
    def initialize_parameters(self, client_manager):
        return self.params

    # ---------- PATCH: return FitIns object ----------
    def configure_fit(self, server_round, parameters, client_manager):
        client = client_manager.sample(1)[0]
        fit_ins = fl.common.FitIns(parameters, {})
        print(f"[{datetime.now():%H:%M:%S}] Round {server_round:02d}: "
              f"send → client {client.cid}")
        return [(client, fit_ins)]

    # --------------------------------------------------
    def aggregate_fit(self, server_round, results, failures):
        if failures:
            print("⚠️  failures:", failures)
        for _, res in results:
            self.buffer.append(
                (fl.common.parameters_to_ndarrays(res.parameters),
                 res.num_examples)
            )
            print(f"    ↳ got update ({len(self.buffer)}/{self.B})")

        if len(self.buffer) < self.B:          # not enough yet
            return self.params, {}

        total = sum(n for _, n in self.buffer)
        agg = None
        for params, n in self.buffer:
            agg = [n * p if agg is None else agg[i] + n * p
                   for i, p in enumerate(params)]
        new_params = [p / total for p in agg]
        self.params = fl.common.ndarrays_to_parameters(new_params)
        print(f"[{datetime.now():%H:%M:%S}] ► aggregated {self.B} updates")
        self.buffer.clear()
        self.last_agg = time.time()
        return self.params, {}

    # no client-side evaluation
    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    # no central dataset
    def evaluate(self, server_round, parameters, config=None):
        print(f"[{datetime.now():%H:%M:%S}] evaluate() called → returning None")
        return None

# ─────────────────────────── 6. Main ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=int, default=5)
    parser.add_argument("--partition", choices=["iid", "labelskew"], default="iid")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--data", type=str, default="datasets/train.csv")
    args = parser.parse_args()

    # Load and split data
    Xtr, Xte, ytr, yte = load_dataframe(args.data)
    parts = split_iid(Xtr, ytr) if args.partition == "iid" else split_label_skew(Xtr, ytr)
    shards = [(Xtr[idx], ytr[idx]) for idx in parts]
    d_in, n_cls = Xtr.shape[1], len(np.unique(ytr))

    # Client factory for simulation
    def client_fn(cid: str):
        # Make every 3rd client slower to simulate stragglers
        slow = (int(cid) % 3 == 0)
        return Client(cid, shards[int(cid)], d_in, n_cls, slow)

    # Initialize FedBuff strategy with specified buffer size
    strategy = FedBuff(buffer_size=args.buffer, d_in=d_in, n_cls=n_cls)

    print(f"\n🟢 Starting FedBuff: {len(shards)} clients | buffer={args.buffer} | "
          f"{args.partition.upper()} | rounds={args.rounds}")
    # Start the Flower simulation (note: start_simulation is deprecated in new versions of Flower)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(shards),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    # After simulation, retrieve the final global model
    model = CreditNet(d_in, n_cls)
    final_params = fl.common.parameters_to_ndarrays(strategy.params)
    model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), final_params)}, strict=True)
    model.eval()
    with torch.no_grad():
        # Compute global model accuracy on the held-out test set
        logits = model(torch.tensor(Xte, dtype=torch.float32))
        accuracy = (logits.argmax(1).numpy() == yte).mean()

    # Save the global model and preprocessing artifacts
    torch.save(model.state_dict(), "fedbuff_model.pth")
    np.save("fedbuff_scaler.npy", {"mean": scaler.mean_, "scale": scaler.scale_})
    with open("fedbuff_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\n✅ Global test accuracy: {accuracy:.4f}")
    print("💾 Saved: fedbuff_model.pth, fedbuff_scaler.npy, fedbuff_cols.txt")

if __name__ == "__main__":
    start = time.time()
    main()
