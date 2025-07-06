import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Net
from dataProcessing import make_client_splits, load_and_preprocess
import random
import time

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_ds, val_ds, local_epochs=1, is_async=False):
        self.model = model
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = local_epochs
        self.is_async = is_async

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters])
        state_dict = {k: v for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Simulate async client taking random time before training
        if self.is_async:
            delay = random.uniform(2, 5)  # Async clients take 2-5 seconds delay
            print(f"Async client delaying for {delay:.2f} seconds")
            time.sleep(delay)

        self.model.train()
        for _ in range(self.epochs):
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                preds = self.model(X)
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(), len(self.train_loader.dataset), {}

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

from flwr.common import Context

def get_client_fn(splits, local_epochs=1, async_client_ids=None):
    async_client_ids = set(async_client_ids or [])

    def client_fn(context: Context):
        cid = context.run_id
        cid_int = int(cid)
        df_client = splits[cid_int]

        X = torch.tensor(df_client.drop('target', axis=1).values, dtype=torch.float32)
        y = torch.tensor(df_client['target'].values, dtype=torch.long)

        train_size = int(0.8 * len(df_client))
        val_size = len(df_client) - train_size
        train_ds, val_ds = random_split(TensorDataset(X, y), [train_size, val_size])

        num_features = X.shape[1]
        num_classes = len(df_client['target'].unique())
        model = Net(num_features=num_features, num_classes=num_classes)

        is_async = cid in async_client_ids

        return FLClient(model, train_ds, val_ds, local_epochs, is_async=is_async).to_client()

    return client_fn