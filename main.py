import sys
import time
from random import randrange
import random

from ale_python_interface import ALEInterface
#import tensorflow as tf
from MarkovDecisionProcess import MarkovDecisionProcess

seed = 250
ale = ALEInterface()
ale.setInt(b'random_seed', 250)
random.seed(250)


ale.setBool(b'sound', False)
#ale.setBool(b'display_screen', True)

name = "Qbert.a26"
rom = str.encode('roms/' + name)
ale.loadROM(rom)


print(sys.setrecursionlimit(50000))
mdp = MarkovDecisionProcess(name, ale)
mdp.run()
