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
        self.output_activation = nn.Hardtanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.output_activation(x)
        x = self.round(x)
        return x

    def get_weights(self):
        """Return all weights as a single flat tensor."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def load_weights(self, weights_path):
        """Load weights from checkpoint."""
        checkpoint = torch.load(weights_path, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])

    def save_weights(self, path):
        """Save model weights to checkpoint."""
        torch.save({'model_state_dict': self.state_dict()}, path)

    def round(self, input):
        """
        Clamp tensor values to [0, 1] and round to 0 or 1 as int.
        0 <= x < 0.5 -> 0
        0.5 <= x <= 1 -> 1
        """
        rounded = torch.where(input < 0.5, torch.zeros_like(input), torch.ones_like(input))
        return rounded.int()