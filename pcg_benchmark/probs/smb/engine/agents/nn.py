import torch

from ..core import MarioAgent
from ..helper import MarioActions
from .networks.simplenn import SimpleNN
import numpy as np


class Agent(MarioAgent):
    def __init__(self, seed=None, weights_path=None):
        super().__init__(seed)
        self.seed = seed
        if weights_path is not None:
            self.weights_path = weights_path
        else:
            self.weights_path = None

    def initialize(self, model):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # send the observation in
        obs = model.getScreenCompleteObservation()
        obs = np.array(obs, dtype=np.float32)
        print(type(obs))
       
        # hidden should be an attribute of this model. Right now it defaults to 128
        output_size = MarioActions.numberOfActions()

        self.brain = SimpleNN(obs=obs, output_size=output_size)
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
