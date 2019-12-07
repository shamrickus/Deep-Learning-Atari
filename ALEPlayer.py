import time
from random import randrange
import random

import BaseROM
from ALEWrapper import ALEWrapper


class ALEPlayer:
    ale = None
    reward = 0
    actions = []
    actionHistory = []
    time = None
    depth = 0
    originalState = None
    rom = None

    def __init__(self, rom: BaseROM):
        self.rom = rom
        self.ale = ALEWrapper()
        self.ale.set_repeat_action()
        self.ale.set_seed(random.randint(0, 99999))
        self.ale.load_rom(rom)
        self.actions = self.ale.get_minimal_actions()
        self.originalState = self.ale.copy_system_state()

    def __str__(self):
        return "reward: {}, depth: {}, gameover: {}, state: {}".format(self.reward, self.depth, self.isGameOver(False), self.originalState)

    def act(self):
        return

    def run(self):
        return
