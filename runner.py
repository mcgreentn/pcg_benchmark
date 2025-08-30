from pcg_benchmark.probs.smb.engine import runLevel

# TODO we have to have a list of mario levels readily available in a folder somewhere
# load them into the runner as text
levelPath = "./data/smb/original/lvl-1.txt"
# read this
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