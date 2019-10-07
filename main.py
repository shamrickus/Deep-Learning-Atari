import sys
from random import randrange

from ale_python_interface import ALEInterface
#import tensorflow as tf

ale = ALEInterface()
ale.setInt(b'random_seed', 123)


ale.setBool(b'sound', True)
ale.setBool(b'display_screen', True)

rom = str.encode('roms/Pitfall.a26')
ale.loadROM(rom)

actions = ale.getLegalActionSet()


reward = 0
while not ale.game_over():
    a = actions[randrange(len(actions))]
    reward += ale.act(a)
print(reward)
