import sys
import time
from random import randrange

from ale_python_interface import ALEInterface
#import tensorflow as tf
from MarkovDecisionProcess import MarkovDecisionProcess

ale = ALEInterface()
ale.setInt(b'random_seed', 123)


ale.setBool(b'sound', False)
#ale.setBool(b'display_screen', True)

rom = str.encode('roms/Breakout.a26')
ale.loadROM(rom)


print(sys.setrecursionlimit(50000))
mdp = MarkovDecisionProcess("Breakout.a26", ale)
mdp.run()
