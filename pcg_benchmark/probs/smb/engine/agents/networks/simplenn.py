import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_out_shape(input_dim, height, width, out_dim, k_height, k_width, padding=0, stride=1, dilation=1):
    height = int((height + 2 * padding - dilation * (k_height - 1) - 1) / stride + 1)
    width = int((width + 2 * padding - dilation * (k_width - 1) - 1) / stride + 1)
    return (out_dim, height, width)

class SimpleNN(nn.Module):
    """
    Simple neural network for Super Mario agent.
    Designed for evolutionary updates (no backpropagation).
    """
    def __init__(self, obs, output_size):
        super().__init__()
        obs_shape = obs.shape
        current_shape = obs_shape
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2)
        current_shape = conv_out_shape(current_shape[0], current_shape[1], current_shape[2], 32, 3, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        current_shape = conv_out_shape(current_shape[0], current_shape[1], current_shape[2], 64, 3, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        current_shape = conv_out_shape(current_shape[0], current_shape[1], current_shape[2], 64, 3, 3, stride=2)
        self.fc1 = nn.Linear(np.prod(current_shape), 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
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
    
    def get_param(self):
        output = np.zeros(self.get_param_size())
        currentIndex = 0
        for name in self._modules:
            mod = self._modules[name]
            weight = mod.weight.data.flatten().numpy()
            bias = mod.bias.data.flatten().numpy()
            output[currentIndex:currentIndex + len(weight)] = weight
            currentIndex += len(weight)
            output[currentIndex:currentIndex + len(bias)] = bias
            currentIndex += len(bias)
        return output
    
    def set_param(self, param):
        currentIndex = 0
        for name in self._modules:
            mod = self._modules[name]
            w_shape = mod.weight.data.shape
            b_shape = mod.bias.data.shape
            mod.weight.data = torch.tensor(param[currentIndex:currentIndex + np.prod(w_shape)].reshape(w_shape)).float()
            currentIndex += np.prod(w_shape)
            mod.bias.data = torch.tensor(param[currentIndex:currentIndex + np.prod(b_shape)].reshape(b_shape)).float()
            currentIndex += np.prod(b_shape)
            
    def get_param_size(self):
        total_param = 0
        for name in self._modules:
            mod = self._modules[name]
            total_param += np.prod(mod.weight.data.shape)
            total_param += np.prod(mod.bias.data.shape)
        return total_param
    def round(self, input):
        """
        Clamp tensor values to [0, 1] and round to 0 or 1 as int.
        0 <= x < 0.5 -> 0
        0.5 <= x <= 1 -> 1
        """
        rounded = torch.where(input < 0.5, torch.zeros_like(input), torch.ones_like(input))
        return rounded.int()