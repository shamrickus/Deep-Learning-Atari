import json
import math
import os
import sys
import time
from json import JSONEncoder
from random import randrange
from statistics import mean

import pydotplus
import jsonpickle
from ale_python_interface import ALEInterface

from ALEPlayer import ALEPlayer

class Node:
    terminal = False
    visits = 0
    depth = 0
    totalReward = 0
    children = None
    state = None
    parent = None
    selectWeight = 0

    def __init__(self, actionSet, depth, parent=None, action=None):
        self.children = dict()
        for action in actionSet:
            self.children[action] = None
        self.depth = depth
        if parent:
            self.parent = parent
        self.action = action

    def __str__(self):
        return "state: {}, depth: {}, reward: {}, visits: {}, weight: {}".format(self.state, self.depth, self.totalReward,
                                                                     self.visits, self.selectWeight)

    def expand(self):
        for k in self.children.keys():
            if self.children[k] is None:
                self.children[k] = Node(self.children.keys(), self.depth + 1, self, k)

    def simulate(self, ale: ALEInterface, depthLimit):
        self.visits += 1
        depth = 0
        self.state = ale.cloneState()
        self.terminal = ale.game_over()
        actionSet = ale.getLegalActionSet()
        while not ale.game_over():
            if depth > depthLimit:
                break
            action = self.selectRandomAction(actionSet)
            reward = ale.act(action)
            self.totalReward += reward*100
            depth += 1
        ale.restoreState(self.state)
        return self.totalReward
        # print(self)
        # print("took: {}".format(time.time() - timer))

    def backPropagate(self, score):
        if self.parent:
            self.parent.update(score)

    def calculateWeight(self):
        if self.parent and self.visits > 0:
            try:
                self.selectWeight = self.totalReward / self.visits + .01*(math.log(self.parent.visits)/self.visits)**(1/2)
            except ValueError:
                pass
            except ZeroDivisionError:
                print("asdf")
        return self.selectWeight

    def update(self, reward):
        self.visits += 1
        self.totalReward += reward
        self.backPropagate(reward)

    def isLeaf(self):
        return self.visits == 0 and self.state == None

    def selectRandomAction(self, actions):
        return actions[randrange(len(actions))]

    def selectBestChild(self):
        children = sorted([x for x in list(self.children.values()) if x is not None and not x.terminal], reverse=True, key=lambda x: x.calculateWeight())
        if len(children) == 0:
            return None
        children = [x for x in children if x.selectWeight == children[0].selectWeight]
        return children[randrange(len(children))]

    def select(self, maxDepth):
        if self.isLeaf():
            return self
        elif self.terminal:
            print("lel")
        #elif self.depth >= maxDepth:
        #    return None
        else:
            for k in self.children.keys():
                if self.children[k] is None:
                    self.children[k] = Node(self.children.keys(), self.depth + 1, self, k)
                if self.children[k].isLeaf():
                    return self.children[k]
            return self.selectBestChild().select(maxDepth)


class MarkovDecisionProcess(ALEPlayer):
    immediateGameover = None
    rootNode: Node = None
    maxTrajectories = 10000
    maxDepth = 300
    exploration = 1

    def __init__(self, rowName, ale):
        super().__init__(rowName, ale)
        self.rootNode = Node(self.actions, 0)


    def fileName(self):
        return "{}.{}".format(self.romName, self.ale.getInt(b'random_seed')) + '.json'

    def preRun(self, outfile=False):
        self.rootNode.expand()
        trajectory = 0
        depth = self.rootNode.depth + 1
        t2 = time.time()
        t = time.time()
        maxDepth = 0
        while trajectory < self.maxTrajectories:
            nodeToRun = self.rootNode.select(depth)
            nodeToRun.expand()
            score = nodeToRun.simulate(self.ale, self.maxDepth)
            nodeToRun.backPropagate(score)
            trajectory += 1
            if nodeToRun.depth > maxDepth:
                maxDepth = nodeToRun.depth
            if trajectory % 100 == 0:
                print("total: {0:.3g}s, {1:.3g}s, {2:.3g}%".format(time.time() - t2, time.time() - t, trajectory / self.maxTrajectories * 100))
                print(maxDepth)
                t = time.time()
        if outfile:
            with open(self.fileName(), 'w') as f:
                f.write("{},{}{}".format(self.maxTrajectories, self.maxDepth, os.linesep))
                f.write(jsonpickle.encode(self.rootNode))
            if self.maxTrajectories <= 500:
                print("Writing to file")
                self.toGraphviz()
        print("done {}, max depth: {}".format(time.time() - t2, maxDepth))

    def run(self):
        #if os.path.isfile(self.fileName()):
        #    with open(self.fileName(), 'r') as f:
        #        self.rootNode = jsonpickle.decode(f.readlines()[1])
        #else:
        rewards = []
        for i in range(0, 1):
            self.ale.reset_game()
            self.preRun(True)
            self.ale.restoreState(self.rootNode.state)
            node = self.rootNode
            actions = self.ale.getLegalActionSet()
            reward = 0
            i = 0
            actionsRan = []
            while not self.ale.game_over():
                child = node.selectBestChild()
                if child is None:
                    action = node.selectRandomAction(actions)
                else:
                    i+= 1
                    action = child.action
                    node = child
                print(action)
                actionsRan.append(action)
                reward +=self.ale.act(action)
            print(i)
            print(jsonpickle.encode(actionsRan))
            print(reward)
            rewards.append(reward)
        print(mean(rewards))
        print(max(rewards))
        print(sum(rewards)/len(rewards))

    def makeGraphizNode(self, node):
        return pydotplus.Node(name="{}".format(node.state), shape='box', style='rounded',
                           label="{}/{}".format(node.totalReward, node.visits))

    def drawNode(self, parent, node):
        if node.isLeaf():
            return
        graph = pydotplus.Subgraph()
        for childKey in [x for x in node.children.keys() if not node.children[x].isLeaf()]:
            child = node.children[childKey]
            n = self.makeGraphizNode(child)
            graph.add_node(n)
            graph.add_edge(pydotplus.Edge(src=parent, dst=n, label=childKey))
            graph.add_subgraph(self.drawNode(n, child))
        return graph

    def toGraphviz(self):
        graph = pydotplus.Dot()
        o = self.makeGraphizNode(self.rootNode)
        graph.add_node(o)
        graph.add_subgraph(self.drawNode(o, self.rootNode))
        graph.write("out.{}.png".format(self.ale.getInt(b'random_seed')), format="png")

    def act(self):
        newState = self.ale.cloneSystemState()
        bestReward = -1
        acts = dict()
        for action in self.actions:
            self.ale.restoreSystemState(newState)
            if action in acts and acts[action]:
                continue
            mdp = MarkovDecisionProcess(self.romName, self.ale)
            mdp.copy(self)
            mdp.runClone(action)
            acts[action] = mdp.immediateGameover
            if mdp.reward > bestReward:
                bestReward = mdp.reward

        if len(set(acts.values())) == 1 and list(acts.values())[0]:
            self.immediateGameover = True
        self.reward += bestReward
        if self.reward > 0:
            print("yo");
