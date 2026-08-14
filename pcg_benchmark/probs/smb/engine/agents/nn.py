import torch

from ..core import MarioAgent
from ..helper import MarioActions
from .networks.simplenn import SimpleNN
import numpy as np


class Agent(MarioAgent):
    """
    Mario agent driven by a SimpleNN. The network is either randomly initialized,
    loaded from a torch checkpoint (weights_path), or seeded from a flat parameter
    vector (weights) as produced by an evolutionary search.
    """
    def __init__(self, seed=None, weights_path=None, weights=None):
        super().__init__(seed)
        self.seed = seed
        self.weights_path = weights_path
        self.weights = weights
        self.brain = None

    def initialize(self, model):
        # MarioGame.setup() calls initialize() on every run, so building the brain has
        # to be idempotent -- otherwise a freshly constructed network would silently
        # replace weights that were assigned between setup() and runGame().
        if self.brain is None:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            # send the observation in
            obs = model.getScreenCompleteObservation()
            obs = np.array(obs, dtype=np.float32)
            # convert to tensor (necessary?)
            obs = torch.tensor(obs, dtype=torch.float32)
            # reshape obs to (1, 16, 16)
            obs = obs.unsqueeze(0)
            output_size = MarioActions.numberOfActions()

            self.brain = SimpleNN(obs=obs, output_size=output_size)

        # Re-apply the requested weights on every initialize so that a replayed game
        # always evaluates the same network.
        if self.weights is not None:
            self.brain.load_weights(self.weights)
        elif self.weights_path:
            self.brain.load_weights(self.weights_path)
        return
    
    def getActions(self, model):
        # get model observation for input
        obs = model.getScreenCompleteObservation()
        # convert to tensor (necessary?)
        obs = torch.tensor(obs, dtype=torch.float32)
        # reshape obs to (1, 16, 16)
        obs = obs.unsqueeze(0)
        
        # output from the model should be a sigmoid [0, 1, 0, 0, 1...]
        output = self.brain(obs)
        output = output.detach().numpy()
        output = output.round().astype(int)
        return output
    
    def getAgentName(self):
        return "MarioNeuralNetAgent"

    def save_weights(self, path):
        # Delegate so the checkpoint layout stays in sync with SimpleNN.load_weights,
        # which expects a dict under the "model_state_dict" key.
        self.brain.save_weights(path)
