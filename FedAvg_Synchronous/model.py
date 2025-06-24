import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_features, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )
    def forward(self, x):
        return self.net(x)
