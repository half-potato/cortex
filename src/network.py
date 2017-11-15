import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import random
import draw_network
from neuron import Neuron

plt.ion()

n = 5
learning_rate = 0.2
punishment_rate = 0.001
cell_death_threhold = 12 # less than this number
cell_death_record = 100
enable_apoptosis = False
PUNISH_DURATION = 1
REWARD_DURATION = 1

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Network:
    def __init__(self, n):
        # each col represents a
        self.weights = np.random.rand(n, n)
        self.neurons = [Neuron() for i in range(n)]
        self.firings = np.zeros((n, 1))
        self.size = n
        self.fire_record = np.zeros((n, 1))
        self.step = 0

        self.randomlyStimulate()
        self.G = nx.DiGraph()
        for i in range(self.size):
            for j in range(self.size):
                w = self.weights[i, j]
                self.G.add_edge(i+1, j+1, weight=w)
        self.edges = self.G.edges()

        cv2.namedWindow("Potentials", 0)
        cv2.resizeWindow("Potentials", 500, 500)

    def update(self):
        # weighted input to neurons
        changes = np.dot(self.weights, self.firings) * 70
        soft = softmax(changes)

        # input to neurons
        rep_x = np.tile(self.firings, (1, self.size))

        self.firings = np.zeros((self.size, 1))
        # update
        pots = np.zeros((self.size, 1), np.float32)
        for i, v in enumerate(self.neurons):
            pots[i,0] = self.neurons[i].potential
            self.firings[i, 0] = self.neurons[i].update(soft[i])

        #print(pots)
        print(np.sum(self.firings))
        # output of neurons
        rep_y = np.tile(self.firings, (1, self.size))

        # Update weights
        dw = learning_rate * rep_y * (rep_x - rep_y*self.weights)
        self.weights += dw
        print("Loss: %f" % np.mean(dw))

        # Kill over firing cells
        self.fire_record += self.firings
        if enable_apoptosis:
            kill_list = np.where(self.fire_record >= cell_death_threhold)
            print(np.max(self.fire_record))
            self.fire_record[kill_list] = 0
            self.weights[:, kill_list[0]] = 0
            if self.step % cell_death_record == 0:
                self.fire_record[np.where(self.fire_record > 0)] -= 1

        # Log
        self.step += 1

    # Add to neurons around the ring
    # x = array of potential differences
    # index = array of indexes of firings
    def input(self, x, index):
        assert(index.shape == x.T.shape)
        self.firings[index] = x

    # Set the output firing of the network
    # y = array of target firings
    # index = array of indexes of firings
    def output(self, y, index):
        assert(np.squeeze(index).shape == np.squeeze(y.T).shape)
        fs = self.firings[np.squeeze(index)]
        differences = np.logical_xor(fs, y)
        punishable = np.logical_and(fs, np.logical_not(y))
        print(np.sum(punishable))
        self.punish(np.sum(punishable)/5, PUNISH_DURATION)
        #similarities = np.logical_not(differences)
        #self.punish(-similarities, REWARD_DURATION)
        # This should automatically strengthen the connections
        self.firings[np.squeeze(index)] = y
        return fs

    def punish(self, error, duration):
        for i, v in enumerate(self.neurons):
            if v.lastFiring < duration:
                change = error * punishment_rate
                # Change it's outbound connections
                self.weights[i, :] -= change
                # Change it's inbound connections
                self.weights[:, i] -= change

    def randomlyStimulate(self):
        print("Randomly stimulated")
        if np.sum(self.firings) < self.size / 8:
            randomFires = self.size // 8
            print(randomFires)
            self.firings = [1] * randomFires + [0] * (self.size - randomFires)
            np.random.shuffle(self.firings)
            self.firings = np.array(self.firings)[np.newaxis, :].T

    def draw(self):
        val_map = {}
        for i, v in enumerate(self.neurons):
            val_map[i+1] = self.neurons[i].potential / 3 + 1
        values = [val_map.get(node, 1) for node in self.G.nodes()]
        #values = [random.random() for node in self.G.nodes()]

        pos = nx.circular_layout(self.G)
        #colors = [G[u][v]["color"] for u, v in edges]
        #weights = [G[u][v]["weight"] for u, v in edges]
        #nx.draw(G, pos, edges=edges, width=weights, arrows=True, cmap=plt.get_cmap("jet"), node_color=values)
        nx.draw_networkx_nodes(self.G, pos, cmap=plt.get_cmap("jet"),
                               node_color=values, node_size=500)
        #nx.draw_networkx_edges(self.G, pos, edges=self.edges,
                #arrows=True)
        plt.pause(0.05)

    def display(self):
        h = int(math.ceil(math.sqrt(self.size)))
        w = int(math.ceil(float(self.size) / h))
        img = np.zeros((h, w))
        for i, v in enumerate(self.neurons):
            y = int(math.floor(float(i) / w))
            x = i - y*w
            img[y, x] = v.potential
        cv2.imshow("Potentials", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    net = Network(2000)
    while True:
       net.update()
       #net.draw()
       net.display()
