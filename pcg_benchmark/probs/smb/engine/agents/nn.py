import torch

from ..core import MarioAgent
from ..helper import MarioActions
from .networks.simplenn import SimpleNN


class Agent(MarioAgent):
    def __init__(self, seed=None, weights_path=None):
        super().__init__(seed)
        self.seed = seed
        if weights_path is not None:
            self.weights_path = weights_path

    def initialize(self, model):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Flatten the 2D observation to 1D for neural network input sizing
        obs = model.getScreenCompleteObservation()
        obs_flat = [item for sublist in obs for item in sublist]
        input_size = len(obs_flat)
        total_size = input_size
        # hidden should be an attribute of this model. Right now it defaults to 128
        hidden_size = getattr(self, 'hidden_size', 128)
        output_size = MarioActions.numberOfActions()
        total_size += hidden_size + output_size
        self.total_size = total_size

        self.brain = SimpleNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        if self.weights_path:
            print(f"Loading weights from {self.weights_path}")
            self.brain.load_weights(self.weights_path)
        else:
            self.brain.save_weights("./data/smb/weights.json")  # Save initial weights
        return
    
    def getActions(self, model):
        # get model observation for input
        obs = model.getScreenCompleteObservation()
        obs_flat = [item for sublist in obs for item in sublist]
        # convert to tensor (necessary?)
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32)
        
        # output from the model should be a sigmoid [0, 1, 0, 0, 1...]
        output = self.brain(obs_tensor)
        print(f"NN raw output: {output}")

        return output
    
    def getAgentName(self):
        return "MarioNeuralNetAgent"
    

    def save_weights(agent, path):
        torch.save(agent.brain.state_dict(), path)
