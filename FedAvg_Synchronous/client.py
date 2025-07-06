# fedbuff_client.py  (drop-in replacement for FedBuff runs)
import time
import random
import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Net                          # your MLP
from dataProcessing import make_client_splits  # if you need it
                                               # load_and_preprocess likewise

# ─────────────────────────── Helper ────────────────────────────
def maybe_sleep(factor: float = 1.0):
    """Random pause to emulate stragglers (optional)."""
    if random.random() < 0.33:                 # ~33 % of clients are slow
        time.sleep(factor * random.uniform(1, 3))

# ─────────────────────────── Client ────────────────────────────
class FedBuffClient(fl.client.NumPyClient):
    """Client class usable with the asynchronous FedBuff strategy."""

    def __init__(self, model, train_ds, val_ds, local_epochs: int = 1):
        self.model = model
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=32)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = local_epochs

    # -------- parameter helpers --------
    def get_parameters(self, config=None):
        return [p.cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    # -------- training (called by FedBuff server) --------
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        maybe_sleep()                          # emulate heterogeneity
        self.model.train()
        for _ in range(self.epochs):
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step()

        # Return weights, number of local examples, and optional metrics
        num_examples = len(self.train_loader.dataset)
        metrics = {"samples": num_examples}     # FedBuff can ignore or use
        return self.get_parameters(), num_examples, metrics

    # -------- (rarely used in pure FedBuff) --------
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in self.val_loader:
                preds = self.model(X)
                loss += float(self.criterion(preds, y)) * len(y)
                correct += int((preds.argmax(1) == y).sum())
                total += len(y)
        return loss / total, total, {"accuracy": correct / total}

# ────────────────────── factory for Flower simulation ───────────────────────
def get_fedbuff_client_fn(splits, local_epochs: int = 1):
    """
    Returns a client_fn compatible with Flower's start_simulation
    but tailored for asynchronous FedBuff.
    """
    def client_fn(cid: str):
        idx = int(cid)
        df_local = splits[idx]

        # build tensors
        X = torch.tensor(df_local.drop('target', axis=1).values, dtype=torch.float32)
        y = torch.tensor(df_local['target'].values,          dtype=torch.long)

        # 80 / 20 split for train / val
        train_size = int(0.8 * len(df_local))
        val_size   = len(df_local) - train_size
        train_ds, val_ds = random_split(TensorDataset(X, y), [train_size, val_size])

        # create model with correct dimensions
        model = Net(num_features=X.shape[1], num_classes=len(df_local['target'].unique()))

        # return FedBuff-capable client
        return FedBuffClient(model, train_ds, val_ds, local_epochs)

    return client_fn
