import os
import csv
import networkx as nx
import time
from kmeans import k_means


def load_graph():
    os.chdir("../facebook_large")

    G = nx.Graph()

    with open('musae_facebook_edges.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G


start = time.time()

G = load_graph()
k_means(G)

print(time.time()-start)