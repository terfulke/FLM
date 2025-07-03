import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Net
from dataProcessing import make_client_splits, load_and_preprocess


class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_ds, val_ds, local_epochs: int = 1):
        self.model = model
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = local_epochs

    def get_parameters(self, config=None):
        # Return model parameters as a list of NumPy arrays
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # Set model parameters from a list of NumPy arrays
        params_dict = zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters])
        state_dict = {k: v for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Update model, train locally for `self.epochs` epochs
        self.set_parameters(parameters)
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
        # Evaluate model on local validation data
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


def get_client_fn(splits, local_epochs: int = 1):
    """
    Factory to create a client_fn for Flower simulation.

    Args:
        splits: List of DataFrames, one per client.
        local_epochs: Number of epochs for each client to train.

    Returns:
        A function that takes a client ID (str) and returns an FLClient.
    """
    def client_fn(cid: str):
        cid_int = int(cid)
        df_client = splits[cid_int]
        # Prepare tensors
        X = torch.tensor(df_client.drop('target', axis=1).values, dtype=torch.float32)
        y = torch.tensor(df_client['target'].values, dtype=torch.long)
        # Train/val split
        train_size = int(0.8 * len(df_client))
        val_size = len(df_client) - train_size
        train_ds, val_ds = random_split(TensorDataset(X, y), [train_size, val_size])
        # Build model
        num_features = X.shape[1]
        num_classes = len(df_client['target'].unique())
        model = Net(num_features=num_features, num_classes=num_classes)
        # Return a client configured with local_epochs
        return FLClient(model, train_ds, val_ds, local_epochs)

    return client_fn
