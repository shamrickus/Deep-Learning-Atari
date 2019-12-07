import os
import time
import random

import BaseROM
from ALEPlayer import ALEPlayer
from Node import Node
from NN import NN, HybridNN, ClassificationNN, RegressionNN


class MarkovDecisionProcess(ALEPlayer):
    rootNode: Node = None
    maxTrajectories = 10000
    maxDepth = 300
    runs = 40
    models = []
    squeeze_ram = False

    def __init__(self, rom: BaseROM, squeeze_ram=False):
        super().__init__(rom)
        self.ale.act(0)
        self.rootNode = Node(self.actions, 0, self.ale)
        self.models = [RegressionNN(rom), ClassificationNN(rom), HybridNN(rom)]
        self.squeeze_ram = squeeze_ram

    def solve(self):
        self.rootNode.expand()
        trajectory = 0
        t2 = time.time()
        t = time.time()
        while trajectory < self.maxTrajectories:
            nodeToRun = self.rootNode.select()
            nodeToRun.expand()
            score = nodeToRun.simulate(self.ale, self.maxDepth, self.rom)
            nodeToRun.back_propagate(score)
            trajectory += 1
            if trajectory % 100 == 0:
                print("total: {0:.5g}s, {1:.3g}s, {2:.3g}%".format(time.time() - t2, time.time() - t,
                                                                   trajectory / self.maxTrajectories * 100))
                t = time.time()
        print("done {}".format(time.time() - t2))

    def run_uct(self):
        node = self.rootNode
        actionsRan = []
        switch = False
        frame = 0
        reward = 0
        while not self.ale.game_over():
            child = node.selectBestChild()
            if child is None:
                if not switch:
                    print("Switching to random on frame: {}, score at this point: {}".format(frame, reward))
                    switch = True
                action = node.selectRandomAction()
            else:
                action = child.action
                node = child
            actionsRan.append(int(action))
            reward += self.ale.act(action)
            frame += 1

    def save(self):
        with open("out/{}.mdp".format(self.rom.name), 'w') as f:
            f.write("{},{},{}{}".format(self.maxTrajectories, self.maxDepth, self.runs, os.linesep))
            f.write(self.rootNode.out())
        if self.maxTrajectories <= 500:
            print("Writing to file")
            self.toGraphviz()

    def load(self):
        if os.path.exists("out/{}.mdp".format(self.rom.name)):
            with open("out/{}.mdp".format(self.rom.name), 'w') as f:
                self.rootNode.load(f.readlines()[1:])
            return True
        return False

    # @profile
    def run(self):
        for i in range(0, self.runs):
            print("---------------------RUN: {}".format(i))
            self.ale.reset()
            self.solve()

    def train(self):
        for network in self.models:
            if not network.load_model():
                network.build_model()
                network.build_data_from_root(self.rootNode)
                network.split_data()
                network.train()
                l, a = network.test()
                network.save_model()
                print(f"Training on {network.name()}, loss: {l}, accuracy: {a}")
            else:
                print(f"Loaded model {network.name()}")

    def simulate(self):
        self.ale.set_display(True)
        self.ale.set_seed(random.randint(1, 99999))
        for network in self.models:
            self.ale.set_recording(f"{self.rom.name.split('.')[0]}/{network.name()}")
            self.ale.load_rom(self.rom)
            self.ale.act(1)

            lastFrames = [self.rom.process_image(self.ale.get_frame())]
            lastFrames.append(lastFrames[0])
            lastFrames.append(lastFrames[0])
            lastFrames.append(lastFrames[0])
            reward = 0
            while not self.ale.game_over():
                action = network.predict(lastFrames).argmax()
                if random.random() < .1:
                    action = random.randrange(0, 5)
                reward += self.ale.act(action)
                lastFrames.append(self.rom.process_image(self.ale.get_frame()))
                lastFrames = lastFrames[1:]
            print(f"Reward for network {network.name()}: {reward}")

    def makeGraphizNode(self, node):
        import pydotplus
        return pydotplus.Node(name="{}".format(node.state), shape='box', style='rounded',
                              label="{}/{}".format(node.totalReward, node.visits))

    def drawNode(self, parent, node):
        import pydotplus
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
        import pydotplus
        graph = pydotplus.Dot()
        o = self.makeGraphizNode(self.rootNode)
        graph.add_node(o)
        graph.add_subgraph(self.drawNode(o, self.rootNode))
        graph.write("out.png", format="png")
