import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    """
    Simple neural network for Super Mario agent.
    Designed for evolutionary updates (no backpropagation).
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.output_activation(x)
        return x

    def get_weights(self):
        """Return all weights as a single flat tensor."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_weights(self, flat_weights):
        """Set all weights from a single flat tensor."""
        idx = 0
        for p in self.parameters():
            numel = p.data.numel()
            p.data.copy_(flat_weights[idx:idx+numel].view_as(p.data))
            idx += numel
