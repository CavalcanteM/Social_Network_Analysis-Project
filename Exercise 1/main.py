import os
import csv
import networkx as nx
import time
from kmeans import k_means, parallel_k_means


def load_graph():
    os.chdir("../facebook_large")

    G = nx.Graph()

    with open('musae_facebook_edges.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G




G = load_graph()


start = time.time()
parallel_k_means(G, 4)

print(time.time()-start)


start = time.time()
k_means(G)

print(time.time()-start)