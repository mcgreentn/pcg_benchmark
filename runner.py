import torch
from pcg_benchmark.probs.smb.engine.core import MarioAgent, MarioForwardModel
from pcg_benchmark.probs.smbtile.engine.core import MarioGame

from pcg_benchmark.probs.smb.engine.agents import nn

def runLevel(levelString, gameTime = 20, iterations = 100, stickyActions = 8, marioState = 0, seed = None):
    MarioAgent.iterations = iterations
    MarioAgent.stickyActions = stickyActions
    game = MarioGame()
    agent = nn.Agent(seed, weights_path="./data/smb/weights.json")
    
    # returns MarioResult
    return game.runGame(agent, levelString, gameTime, marioState)

def runLevelWithNet(levelString, net, gameTime = 20, iterations = 100, stickyActions = 8, marioState = 0, seed = None):
    MarioAgent.iterations = iterations
    MarioAgent.stickyActions = stickyActions
    game = MarioGame()
    agent = nn.Agent(seed)
    game.setAgent(agent)
    game.setup(levelString, 20, 0)
    
    agent.brain.load_weights(net) # load weights from net
    
    # returns MarioResult
    return game.runGame(agent, levelString, gameTime, marioState
                        )
if __name__ == "__main__":
    # load them into the runner as text
    levelPath = "./data/smb/original/lvl-1.txt"
    # read this file
    with open(levelPath, 'r') as file:
        level_1 = file.read()

        result = runLevel(level_1, "nn", gameTime=20, iterations=1, seed=0)    
        print("Game Status:", result.getGameStatus())
        print("Completion %:", result.getCompletionPercentage())
        print("Remaining Time:", result.getRemainingTime())
        print("Mario Mode:", result.getMarioMode())
        print("Total Kills:", result.getKillsTotal())
        print("Kills by Fire:", result.getKillsByFire())
        print("Kills by Stomp:", result.getKillsByStomp())
        print("Kills by Shell:", result.getKillsByShell())
        print("Kills by Fall:", result.getKillsByFall())
        print("Num Jumps:", result.getNumJumps())
        print("Max X Jump:", result.getMaxXJump())
        print("Max Jump Air Time:", result.getMaxJumpAirTime())
        print("Current Lives:", result.getCurrentLives())
        print("Current Coins:", result.getCurrentCoins())
        print("Collected Mushrooms:", result.getNumCollectedMushrooms())
        print("Collected Fireflowers:", result.getNumCollectedFireflower())
        print("Collected Tile Coins:", result.getNumCollectedTileCoins())
        print("Destroyed Bricks:", result.getNumDestroyedBricks())
