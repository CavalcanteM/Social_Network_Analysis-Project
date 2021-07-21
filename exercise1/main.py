import csv
import networkx as nx
import time
from girman_newmann import bwt_cluster_naive, bwt_cluster_parallel, bwt_cluster_sampled, bwt_cluster_performance_based

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
bwt_cluster_naive(G)
#bwt_cluster_parallel(G, 4)
#bwt_cluster_sampled(G, 0.05, 4)

print(time.time()-start)