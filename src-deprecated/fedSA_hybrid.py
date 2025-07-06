# ------------------------------- fedsa_async_credit.py ------------------------------------
"""
FedSA (semi-asynchronous) for real Kaggle credit-score dataset
==============================================================
Run:
    python fedsa_async_credit.py --M 3 --tau0 5 --lambda_lr 0.01 --partition iid
    python fedsa_async_credit.py --M 3 --tau0 5 --lambda_lr 0.01 --partition labelskew
"""
import argparse, random, time, collections
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------ 1. Load and clean your dataset ------------------------------
def load_dataframe():
    df = pd.read_csv("datasets/train.csv")

    # Drop negative or impossible ages
    df = df[df["Age"] > 0]

    # Replace placeholder strings
    df = df.replace(["_", "NA"], np.nan)

    # Fill missing numerics with mean
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Fill other missing with mode
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])

    # Parse 'Credit_History_Age'
    def parse_age(x):
        try:
            yrs, mos = 0, 0
            if "Years" in x: yrs = int(x.split("Years")[0].strip())
            if "Months" in x: mos = int(x.split("and")[-1].split("Months")[0].strip())
            return yrs * 12 + mos
        except:
            return 0

    df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(parse_age)
    df = df.drop(columns=["Credit_History_Age"])

    # Drop IDs, names, SSN etc
    drop_cols = ["ID", "Customer_ID", "Month", "Name", "SSN"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode target
    y = df["Credit_Score"].astype("category").cat.codes
    df = df.drop(columns=["Credit_Score"])

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    # Scale numeric features
    X = StandardScaler().fit_transform(df.values)

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# ------------------------------ 2. IID / non-IID splits ------------------------------
def split_iid(X, y, n_clients=10):
    idx = np.random.permutation(len(X))
    return np.array_split(idx, n_clients)

def split_label_skew(X, y, n_clients=10):
    labels = np.unique(y)
    splits = []
    for cid in range(n_clients):
        lab = labels[cid % len(labels)]
        major = np.where(y == lab)[0][:600]
        minor = np.random.choice(np.where(y != lab)[0], 200, replace=False)
        idx = np.concatenate([major, minor])
        splits.append(idx)
    return splits

# ------------------------------ 3. Tiny MLP ------------------------------
class CreditNet(nn.Module):
    def __init__(self, d_in, n_cls=3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, n_cls)
        )
    def forward(self, x): return self.seq(x)

# ------------------------------ 4. Flower Client ------------------------------
class Client(fl.client.NumPyClient):
    def __init__(self, cid, data, d_in, n_cls):
        self.cid = cid
        (X, y) = data
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.model = CreditNet(d_in, n_cls)

    def _set(self, params):
        state = dict(zip(self.model.state_dict(), [torch.tensor(p) for p in params]))
        self.model.load_state_dict(state, strict=True)

    def _get(self):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def get_parameters(self, _): return fl.common.ndarrays_to_parameters(self._get())

    def fit(self, parameters, cfg):
        self._set(fl.common.parameters_to_ndarrays(parameters))
        lr = cfg["lr"]
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(3):
            opt.zero_grad()
            loss = F.cross_entropy(self.model(self.X), self.y)
            loss.backward()
            opt.step()
        return (fl.common.ndarrays_to_parameters(self._get()),
                len(self.X),
                {"round_started": cfg["rnd"]})

    def evaluate(self, parameters, _):
        self._set(fl.common.parameters_to_ndarrays(parameters))
        with torch.no_grad():
            logits = self.model(self.X)
            loss = F.cross_entropy(logits, self.y).item()
            acc = (logits.argmax(1) == self.y).float().mean().item()
        return loss, len(self.X), {"acc": acc}

# ------------------------------ 5. FedSA strategy ------------------------------
class FedSA(fl.server.strategy.Strategy):
    def __init__(self, M=3, tau0=5, lambda_lr=0.01):
        super().__init__()
        self.M, self.tau0, self.lambda_lr = M, tau0, lambda_lr
        self.params = None
        self.buffer = []
        self.staleness = collections.defaultdict(int)
        self.counts = collections.defaultdict(int)

    def initialize_parameters(self, cm):
        if self.params is None:
            dummy = CreditNet(1)
            self.params = fl.common.ndarrays_to_parameters(
                [v.cpu().numpy() for v in dummy.state_dict().values()]
            )
        return self.params

    def configure_fit(self, rnd, params, cm):
        urgent = [c.cid for c in cm.all().values() if self.staleness[c.cid] >= self.tau0]
        chosen = [cm.all()[urgent[0]]] if urgent else cm.sample(1)
        total = sum(self.counts.values()) or 1
        cfgs = []
        for c in chosen:
            fi = self.counts[c.cid] / total if total else 1/len(cm.all())
            lr = self.lambda_lr / (len(cm.all()) * max(fi, 1e-4))
            cfgs.append((c, {"rnd": rnd, "lr": lr}))
        return cfgs

    def aggregate_fit(self, rnd, results, failures):
        for cid, res in [(r[0].cid, r[1]) for r in results]:
            self.buffer.append((fl.common.parameters_to_ndarrays(res.parameters),
                                res.num_examples, cid))
            self.counts[cid] += 1
            self.staleness[cid] = 0
        for c in self.staleness: self.staleness[c] += 1

        if len(self.buffer) < self.M:
            return self.params, {}

        tot = sum(n for _, n, _ in self.buffer)
        agg = None
        for params, n, _ in self.buffer:
            if agg is None:
                agg = [n * p for p in params]
            else:
                for i, p in enumerate(params):
                    agg[i] += n * p
        new_params = [p / tot for p in agg]
        self.params = fl.common.ndarrays_to_parameters(new_params)
        self.buffer.clear()
        return self.params, {}

    def evaluate(self, rnd, params): return None

# ------------------------------ 6. Run ------------------------------
def main():
    X_train, X_test, y_train, y_test = load_dataframe()

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--tau0", type=int, default=5)
    parser.add_argument("--lambda_lr", type=float, default=0.01)
    parser.add_argument("--partition", choices=["iid", "labelskew"], default="iid")
    args = parser.parse_args()

    if args.partition == "iid":
        parts = split_iid(X_train, y_train, 10)
    else:
        parts = split_label_skew(X_train, y_train, 10)
    shards = [(X_train[idx], y_train[idx]) for idx in parts]

    d_in, n_cls = X_train.shape[1], len(np.unique(y_train))

    def client_fn(cid: str):
        return Client(cid, shards[int(cid)], d_in, n_cls)

    strategy = FedSA(M=args.M, tau0=args.tau0, lambda_lr=args.lambda_lr)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(shards),
        config=fl.server.ServerConfig(num_rounds=40),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    # Final test accuracy:
    final_params = fl.common.parameters_to_ndarrays(strategy.params)
    final_model = CreditNet(d_in, n_cls)
    state = dict(zip(final_model.state_dict(), [torch.tensor(p) for p in final_params]))
    final_model.load_state_dict(state, strict=True)
    final_model.eval()
    with torch.no_grad():
        logits = final_model(torch.tensor(X_test, dtype=torch.float32))
        acc = (logits.argmax(1).numpy() == y_test).mean()
    print(f"\n✅ Final global test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
# ------------------------------------------------------------------------------------------
