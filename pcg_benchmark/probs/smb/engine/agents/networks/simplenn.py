import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_out_shape(height, width, out_dim, k_height, k_width, padding=0, stride=1, dilation=1):
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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
        current_shape = conv_out_shape(current_shape[1], current_shape[2], 16, 2, 2, stride=1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        current_shape = (current_shape[0], current_shape[1] // 2, current_shape[2] // 2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=2, stride=1)
        current_shape = conv_out_shape(current_shape[1], current_shape[2], 16, 2, 2, stride=1)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        current_shape = (current_shape[0], current_shape[1] // 2, current_shape[2] // 2)
        
        self.conv3 = nn.Conv2d(16, 16, kernel_size=2, stride=1)
        current_shape = conv_out_shape(current_shape[1], current_shape[2], 16, 2, 2, stride=1)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        current_shape = (current_shape[0], current_shape[1] // 2, current_shape[2] // 2)
        
        self.fc1 = nn.Linear(np.prod(current_shape), 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

    def get_weights(self):
        """Return all weights as a single flat tensor."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def load_weights(self, weights_path):
        """Load weights from a torch checkpoint (str) or a flat parameter vector (ndarray)."""
        if isinstance(weights_path, str):
            checkpoint = torch.load(weights_path, weights_only=True)
            try:
                self.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                # A checkpoint saved from a different architecture fails here with a wall
                # of per-tensor size mismatches. Say what actually went wrong instead.
                raise RuntimeError(
                    f"Checkpoint '{weights_path}' does not match the current SimpleNN "
                    f"architecture ({self.get_param_size()} parameters). It was most likely "
                    f"saved from an older version of the network; regenerate it with "
                    f"SimpleNN.save_weights() or pass a flat parameter vector instead.\n{e}"
                ) from e
        elif isinstance(weights_path, np.ndarray):
            if len(weights_path) != self.get_param_size():
                raise ValueError(
                    f"Expected a flat parameter vector of length {self.get_param_size()}, "
                    f"got {len(weights_path)}."
                )
            self.set_param(weights_path)
        else:
            raise ValueError("weights_path must be a file path (str) or numpy array")

    def save_weights(self, path):
        """Save model weights to checkpoint."""
        torch.save({'model_state_dict': self.state_dict()}, path)
    
    def get_param(self):
        output = np.zeros(self.get_param_size())
        currentIndex = 0
        for name in self._modules:
            mod = self._modules[name]
            if not hasattr(mod, "weight") or not hasattr(mod, "bias"):
                continue
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
            if not hasattr(mod, "weight") or not hasattr(mod, "bias"):
                continue
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
            if not hasattr(mod, "weight") or not hasattr(mod, "bias"):
                continue
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