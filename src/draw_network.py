import networkx as nx
import matplotlib.pyplot as plt

import numpy as np

def draw_network(net):
    G = nx.DiGraph()
    for i in range(net.size):
        for j in range(net.size):
            w = net.weights[i, j]
            G.add_edge(i+1, j+1, weight=w)
    val_map = {}
    for i, v in enumerate(net.neurons):
        val_map[i+1] = v.potential
    values = [val_map.get(node, 0) for node in G.nodes()]

    pos = nx.circular_layout(G)
    edges = G.edges()
    #colors = [G[u][v]["color"] for u, v in edges]
    weights = [G[u][v]["weight"] for u, v in edges]
    print(edges)
    nx.draw(G, pos, edges=edges)#, width=weights)
    plt.show()
