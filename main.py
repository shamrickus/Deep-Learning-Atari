import sys
from BaseROM import Breakout
import tensorflow
from MarkovDecisionProcess import MarkovDecisionProcess

print(tensorflow.__version__)
sys.setrecursionlimit(50000)
rom = Breakout()
mdp = MarkovDecisionProcess(rom, False)
mdp.run()
mdp.save()
mdp.train()
mdp.simulate()

