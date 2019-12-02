import time
from random import randrange

import numpy
from ale_python_interface import ALEInterface


class ALEPlayer:
    romName = ""
    ale = None
    reward = 0
    actions = []
    actionHistory = []
    time = None
    actions = []
    depth = 0
    originalState = None

    def __init__(self, romName, ale: ALEInterface):
        self.romName = romName
        self.ale = ale
        self.actions = self.ale.getLegalActionSet()
        #self.actions = numpy.roll(self.actions, -1)
        self.originalState = self.ale.cloneSystemState()

    def mapAction(self, action):
        if action == 0:
            return "noop"
        elif action == 1:
            return "fire"
        elif action == 2:
            return "up"
        elif action == 3:
            return "right"
        elif action == 4:
            return "left"
        elif action == 5:
            return "down"
        elif action == 6:
            return "up-right"
        elif action == 7:
            return "up-left"
        elif action == 8:
            return "down-right"
        elif action == 9:
            return "down-left"
        elif action == 10:
            return "up-fire"
        elif action == 11:
            return "right-fire"
        elif action == 12:
            return "left-fire"
        elif action == 13:
            return "down-fire"
        elif action == 14:
            return "up-right-fire"
        elif action == 15:
            return "up-left-fire"
        elif action == 16:
            return "down-right-fire"
        elif action == 17:
            return "down-left-fire"
        elif action == 40:
            return "reset"

    def __str__(self):
        return "reward: {}, depth: {}, gameover: {}, state: {}".format(self.reward, self.depth, self.isGameOver(False), self.originalState)

    def isGameOver(self, output=True):
        if self.ale.game_over() and output:
            print("game over. depth: {}, reward: {}".format(self.depth, self.reward))
        return self.ale.game_over()

    def act(self):
        action = self.actions[randrange(len(self.actions))]
        self.actionHistory.append(action)
        self.reward += self.ale.act(action)

    def run(self):
        self.time = time.time()
        while not self.isGameOver():
            self.act()
        print(time.time() - self.time)
        print(self.reward)
