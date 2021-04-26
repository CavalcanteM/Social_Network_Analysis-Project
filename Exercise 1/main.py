import os
import csv
import networkx as nx
import time
from kmeans import k_means, parallel_k_means, optimized_k_means


def load_graph():

    G = nx.Graph()

    with open('../facebook_large/musae_facebook_edges.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G




G = load_graph()

start = time.time()
optimized_k_means(G)

print(time.time()-start)