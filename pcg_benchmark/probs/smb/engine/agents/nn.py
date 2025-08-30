import torch

from ..core import MarioAgent
from ..helper import MarioActions
from .networks.simplenn import SimpleNN


class Agent(MarioAgent):
    def initialize(self, model, weights_path=None):
        obs = model.getScreenCompleteObservation()
        # Flatten the 2D int array to 1D for neural network input sizing
        obs_flat = [item for sublist in obs for item in sublist]
        input_size = len(obs_flat)
        
        # hidden should be an attribute of this model. Right now it defaults to 128
        hidden_size = getattr(self, 'hidden_size', 128)
        self.brain = SimpleNN(input_size=input_size, hidden_size=hidden_size, output_size=MarioActions.numberOfActions())
        if weights_path:
            # TODO mcgreen: verify that we want CPU :) I think we do since its evo not backprop
            weights = torch.load(weights_path, map_location='cpu')
            self.brain.set_weights(weights)
        return
    
    def getActions(self, model):
        # get model observation for input
        obs = model.getScreenCompleteObservation()
        obs_flat = [item for sublist in obs for item in sublist]
        # convert to tensor (necessary?)
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32)
        
        # output from the model should be a sigmoid [0, 1, 0, 0, 1...]
        output = self.brain(obs_tensor)
        
        return output
    
    def getAgentName(self):
        return "MarioNeuralNetAgent"
    