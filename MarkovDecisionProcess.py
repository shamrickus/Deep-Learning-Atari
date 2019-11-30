import time
from random import randrange

import pydotplus
from ale_python_interface import ALEInterface

from ALEPlayer import ALEPlayer

class Node:
    terminal = False
    visits = 0
    score = 0
    depth = 0
    totalReward = 0
    children = None
    state = None

    def __init__(self, actionSet, depth):
        self.children = dict()
        for action in actionSet:
            self.children[action] = None
        self.depth = depth

    def __str__(self):
        return "state: {}, depth: {}, reward: {}, visits: {}".format(self.state, self.depth, self.totalReward, self.visits)

    def unexploredChildren(self):
        return [(k,v) for k,v in self.children if v is None]

    def notSimulated(self):
        return len([(k,v) for k,v in self.children if v is None or v.score == 0]) == 0

    def expand(self):
        for k in self.children.keys():
            self.children[k] = Node(self.children.keys(), self.depth+1)

    def simulate(self, ale: ALEInterface, depthLimit):
        self.visits += 1
        timer = time.time()
        depth = 0
        self.state = ale.cloneState()
        self.terminal = ale.game_over()
        while not ale.game_over():
            if depth > depthLimit:
                break
            actionSet = ale.getLegalActionSet()
            action = self.selectAction(actionSet)
            reward = ale.act(action)
            self.totalReward += reward
            depth += 1
        ale.restoreState(self.state)
        #print(self)
        #print("took: {}".format(time.time() - timer))


    def isLeaf(self):
        return self.visits == 0 and self.state == None

    def selectAction(self, actions):
        return actions[randrange(len(actions))]

    def select(self, maxDepth):
        if self.isLeaf():
           return self
        elif self.terminal:
            print("lel")
        elif self.depth >= maxDepth:
            return None
        else:
            for k in self.children.keys():
                if self.children[k] is None:
                    self.children[k] = Node(self.children.keys(), self.depth+1)
                if self.children[k].isLeaf():
                    return self.children[k]
            for v in self.children.values():
                valid = v.select(maxDepth)
                if valid is not None:
                    return valid
        return None





class MarkovDecisionProcess(ALEPlayer):
    immediateGameover = None
    rootNode: Node = None
    maxTrajectories = 10000
    maxDepth = 300
    exploration = 1

    def __init__(self, rowName, ale):
        super().__init__(rowName, ale)
        self.rootNode = Node(self.actions, 0)

    def copy(self, alp):
        self.reward = alp.reward
        self.actionHistory = alp.actionHistory
        self.actions = alp.actions
        self.depth = alp.depth + 1

    def runClone(self, action):
        print(self.mapAction(action), self.depth)
        self.ale.act(action)
        while not self.isGameOver():
            self.act()

    def run(self):
        self.rootNode.expand()
        trajectory = 0
        node = self.rootNode
        depth = node.depth+1
        t = time.time()
        while trajectory < 5000:
            nodeToRun = node.select(depth)
            if nodeToRun is None:
                depth+=1
                continue
            nodeToRun.expand()
            nodeToRun.simulate(self.ale, self.maxDepth)
            trajectory+=1
            if trajectory % 100 == 0:
                print("{} sec, {}%".format(time.time() - t, trajectory/5000*100))
                t = time.time()
        print("Writing to file")
        print(nodeToRun.depth)
        self.toGraphviz()
        print("done")

    def drawNode(self, parent, node):
        if node.isLeaf():
            return
        graph = pydotplus.Subgraph()
        for childKey in [x for x in node.children.keys() if not node.children[x].isLeaf()]:
            child = node.children[childKey]
            n = pydotplus.Node(name="{}".format(child.state), shape='box', style='rounded', label=str(child.totalReward))
            graph.add_node(n)
            graph.add_edge(pydotplus.Edge(src=parent, dst=n, label=childKey))
            graph.add_subgraph(self.drawNode(n, child))
        return graph

    def toGraphviz(self):
        graph = pydotplus.Dot()
        node = self.rootNode
        o = pydotplus.Node(name=node.totalReward, shape="box", style="rounded")
        graph.add_node(o)
        graph.add_subgraph(self.drawNode(o, node))
        graph.write("out.png", format="png")

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
