import os

import numpy
import sklearn.model_selection as sk
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python.keras import losses

from BaseROM import BaseROM
from Node import Node


class NN:
    rom: BaseROM = None
    model = None
    x = []
    y = []
    x_test = []
    x_train = []
    y_test = []
    y_train = []

    def __init__(self, rom: BaseROM):
        self.rom = rom

    def build_model(self):
        self.model = Sequential([
            Conv2D(84, (8, 8), strides=4, input_shape=(84, 84, 4)),
            Conv2D(32, (4, 4), strides=2),
            Flatten(),
            Dense(256)
        ])
        self.add_output_layer()
        self.model.compile(optimizer='adam',
                           loss=losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])

    def add_output_layer(self):
        self.model.add(Dense(len(self.rom.actions()), activation='linear'))

    def build_data_from_root(self, rootNode: Node):
        return

    def predict(self, frames):
        frames = numpy.asarray(frames)
        frames = frames.reshape((1, 84, 84, 4))
        return self.model.predict(frames)

    def split_data(self):
        self.x = numpy.asarray(self.x)
        self.y = numpy.asarray(self.y)
        self.x = self.x.reshape((self.x.shape[0], 84, 84, 4))
        self.x_train, self.x_test, self.y_train, self.y_test = sk.train_test_split(self.x, self.y, test_size=.1)

    def train(self):
        self.model.fit(self.x_train, self.y_train, epochs=10)

    def test(self):
        return self.model.evaluate(self.x_test, self.y_test)

    def save_model(self):
        self.model.save(f"models/{self.name()}.{self.rom.name}.h5", save_format="tf")

    def load_model(self):
        if os.path.exists(f"models/{self.name()}.{self.rom.name}.h5"):
            self.model = load_model(f"models/{self.name()}.{self.rom.name}.h5")
            return True
        return False

    def name(self):
        return "NN"


class ClassificationNN(NN):
    def add_output_layer(self):
        self.model.add(Dense(len(self.rom.actions()), activation='softmax'))

    def name(self):
        return "ClassificationNN"

    def build_data_from_root(self, rootNode: Node):
        frames = numpy.asarray(rootNode.last_four_frames())
        self.x.append(frames.reshape((84, 84, 4)))
        bestChild = rootNode.selectBestChild()
        if bestChild is None:
            bestChild = rootNode.selectRandomAction()
        self.y.append(bestChild.action)
        for child in [child for child in rootNode.children.values() if not child.isLeaf()]:
            self.build_data_from_root(child)


class RegressionNN(NN):
    def add_output_layer(self):
        self.model.add(Dense(len(self.rom.actions()), activation='linear'))

    def name(self):
        return "RegressionNN"

    def split_data(self):
        self.x = numpy.asarray(self.x)
        self.y = numpy.asarray(self.y)
        self.x = self.x.reshape(self.x.shape[0], 84, 84, 4)
        self.x_train, self.x_test, self.y_train, self.y_test = sk.train_test_split(self.x, self.y, test_size=.1)

    def build_data_from_root(self, rootNode: Node):
        frames = numpy.asarray(rootNode.last_four_frames())
        self.x.append(frames.reshape((84, 84, 4)))
        bestChild = rootNode.selectBestChild()
        if bestChild is None:
            bestChild = rootNode.selectRandomAction()
        self.y.append(bestChild.selectWeight)
        for child in [child for child in rootNode.children.values() if not child.isLeaf()]:
            self.build_data_from_root(child)


class HybridNN(ClassificationNN):
    x_trains = []
    x_tests = []
    y_trains = []
    y_tests = []
    def name(self):
        return "HybridNN"

    def split_data(self):
        self.x = numpy.asarray(self.x)
        self.y = numpy.asarray(self.y)
        self.x = self.x.reshape((self.x.shape[0], 84, 84, 4))
        xLen = int(self.x.shape[0]/4)
        yLen = int(self.y.shape[0]/4)
        x_train, x_test, y_train, y_test = sk.train_test_split(self.x[0:xLen,...], self.y[0:yLen,...], test_size=.1)
        self.x_trains.append(x_train)
        self.x_tests.append(x_test)
        self.y_trains.append(y_train)
        self.y_tests.append(y_test)

    def train(self):
        self.model.fit(self.x_trains[0], self.y_trains[0], epochs=10)
        xLen = int(self.x.shape[0]/4)
        yLen = int(self.y.shape[0]/4)
        for i in range(0, int(xLen)):
            pred = self.predict(self.x[i].tolist()).argmax()
            self.y[i] = pred
        x_train, x_test, y_train, y_test = sk.train_test_split(self.x[xLen:2*xLen,...], self.y[yLen:2*yLen,...], test_size=.1)
        self.x_trains.append(x_train)
        self.x_tests.append(x_test)
        self.y_trains.append(y_train)
        self.y_tests.append(y_test)
        self.model.fit(self.x_trains[1], self.y_trains[1], epochs=10)
        for i in range(int(xLen), int(2*xLen)):
            pred = self.predict(self.x[i].tolist()).argmax()
            self.y[i] = pred
        x_train, x_test, y_train, y_test = sk.train_test_split(self.x[xLen*2:3*xLen,...], self.y[yLen*2:3*yLen,...], test_size=.1)
        self.x_trains.append(x_train)
        self.x_tests.append(x_test)
        self.y_trains.append(y_train)
        self.y_tests.append(y_test)
        self.model.fit(self.x_trains[2], self.y_trains[2], epochs=10)
        for i in range(int(2*xLen), int(3*xLen)):
            pred = self.predict(self.x[i].tolist()).argmax()
            self.y[i] = pred
        x_train, x_test, y_train, y_test = sk.train_test_split(self.x[3*xLen:,...], self.y[3*yLen:,...], test_size=.1)
        self.x_trains.append(x_train)
        self.x_tests.append(x_test)
        self.y_trains.append(y_train)
        self.y_tests.append(y_test)
        self.model.fit(self.x_trains[3], self.y_trains[3], epochs=10)
        for i in range(int(3*xLen), self.x.shape[0]):
            pred = self.predict(self.x[i].tolist()).argmax()
            self.y[i] = pred
        self.x_train, self.x_test, self.y_train, self.y_test = sk.train_test_split(self.x, self.y, test_size=.1)
