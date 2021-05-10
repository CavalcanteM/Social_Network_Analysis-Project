import csv
import networkx as nx
import time
from hits import dict_hits, matrix_hits, parallel_matrix_hits, parallel_dict_hits
import math

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
matrix_hits(G, 200)

print(time.time()-start)
