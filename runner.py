from pcg_benchmark.probs.smb.engine import runLevel

# TODO we have to have a list of mario levels readily available in a folder somewhere
# load them into the runner as text
levelPath = "./data/smb/original/lvl-1.txt"
# read this
with open(levelPath, 'r') as file:
    level_1 = file.read()

    runLevel(level_1, "nn", gameTime=20, iterations=1, seed=0)    
