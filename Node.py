import math
import os
from random import randrange

import jsonpickle
import numpy

import ALEWrapper
import BaseROM

global_id = 0


class Node:
    terminal = False
    visits = 0
    depth = 0
    totalReward = 0
    children = None
    state = None
    systemState = None
    parent = None
    selectWeight = 0
    actionSet = None
    ale: ALEWrapper = None
    id = None

    def __init__(self, actionSet, depth, ale: ALEWrapper, parent=None, action=None):
        global global_id
        self.children = dict()
        self.ale = ale
        self.actionSet = actionSet
        for k in self.actionSet:
            self.children[k] = None
        self.depth = depth
        if parent:
            self.parent = parent
        self.action = action
        self.id = global_id
        global_id += 1

    def __str__(self):
        return "action: {}, depth: {}, reward: {}, visits: {}, weight: {}".format(self.action, self.depth,
                                                                                  self.totalReward,
                                                                                  self.visits, self.selectWeight)

    # @profile
    def out(self):
        if self.isLeaf():
            return ""
        children = [self.children[k] for k in self.actionSet if
                    self.children[k] is not None and not self.children[k].isLeaf()]
        res = "{}|{}|{}|{}|{}|{}|{}{}".format(self.depth, self.totalReward,
                                              self.action, float(self.selectWeight), self.visits, len(children),
                                              self.id, os.linesep)
        for c in children:
            res += "{}".format(c.out(), os.linesep)
        return res

    def load(self, lines):
        children = 0
        for i, value in enumerate(lines[0].split("|")):
            if i == 0:
                self.depth = int(value)
            elif i == 1:
                self.totalReward = int(value)
            elif i == 2:
                try:
                    self.action = int(value)
                except ValueError:
                    self.action = None
            elif i == 3:
                self.selectWeight = float(value)
            elif i == 4:
                self.visits = int(value)
            elif i == 5:
                children = int(value)
            elif i == 6:
                # self.frame = numpy.asarray(jsonpickle.decode(value))
                self.id = int(value)
        lines = lines[1:]
        self.expand()
        for i in range(0, children):
            action = int(lines[0].split("|")[2])
            lines = self.children[action].load(lines)
        return lines

    def expand(self):
        for k in self.actionSet:
            if self.children[k] is None:
                self.children[k] = Node(self.actionSet, self.depth + 1, self.ale, self, k)

    def simulate(self, ale: ALEWrapper, depthLimit, rom: BaseROM, squeeze_ram=False):
        self.visits += 1
        depth = 0
        reward = 0
        if self.parent is not None:
            ale.restore_state(self.parent.state)
            reward += ale.act(self.action)
        if squeeze_ram:
            with open(os.path.join(os.getcwd(), "out", f"node{self.id}.dat"), 'w') as data:
                data.write(str(rom.process_image(ale.get_frame()).tolist()))
        else:
            self.frame = rom.process_image(ale.get_frame())
        self.state = ale.copy_state()
        self.terminal = ale.game_over()
        while not ale.game_over():
            if depth > depthLimit:
                break
            action = self.selectRandomAction()
            reward += ale.act(action)
            depth += 1
        ale.restore_state(self.state)
        self.totalReward = reward
        return reward

    def max_reward(self):
        reward = self.totalReward
        for child in self.children.values():
            if child is not None:
                tempReward = child.max_reward()
                if tempReward > reward:
                    reward = tempReward
        return reward

    def back_propagate(self, score):
        if self.parent:
            self.parent.update(score)

    def calculateWeight(self):
        if self.parent and self.visits > 0:
            try:
                self.selectWeight = self.totalReward / self.visits + math.sqrt(
                    (math.log(self.parent.visits) / self.visits))
            except ValueError:
                pass
        return self.selectWeight

    def load_frame(self):
        if self.frame is not None:
            return self.frame
        with open(os.path.join(os.getcwd(), "out", f"node{self.id}.dat"), 'r') as f:
            return numpy.asarray(jsonpickle.decode(f.read()))

    def last_four_frames(self):
        frames = [self.load_frame()]
        node = self.parent
        while node is not None and node.depth != 0 and len(frames) < 4:
            frames.append(node.load_frame())
            node = node.parent
        while len(frames) < 4:
            frames.append(frames[-1])
        return frames

    def update(self, reward):
        self.visits += 1
        self.totalReward += reward
        self.back_propagate(reward)

    def isLeaf(self):
        return self.visits == 0 and self.state is None

    def selectRandomAction(self):
        return self.actionSet[randrange(len(self.actionSet))]

    def selectBestChild(self):
        children = sorted([x for x in list(self.children.values()) if x is not None and not x.terminal], reverse=True,
                          key=lambda x: x.calculateWeight())
        if len(children) == 0:
            return None
        children = [x for x in children if x.selectWeight == children[0].selectWeight]
        return children[randrange(len(children))]

    def select(self):
        if self.isLeaf():
            return self
        elif self.terminal:
            return None
        else:
            self.expand()
            leafChildren = [self.children[x] for x in self.actionSet if self.children[x].isLeaf()]
            if len(leafChildren) == 0:
                return self.selectBestChild().select()
            else:
                return leafChildren[randrange(len(leafChildren))]
