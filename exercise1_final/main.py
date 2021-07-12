import csv
import networkx as nx
import time
from shapley import shapley_degree, shapley_threshold, shapley_closeness


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

sp = shapley_degree(G)
k = 1 # parameter for shapley threshold
st = shapley_threshold(G, k)
u = 1 # minimum distance used in shapley closeness
sc = shapley_closeness(G, u)

print(time.time()-start)