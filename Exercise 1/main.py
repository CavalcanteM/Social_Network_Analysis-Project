import csv
import networkx as nx
import time
from kmeans import k_means, parallel_k_means, optimized_k_means_v2, parallel_opt_k_means_v2
from spectral import spectral_invpm, spectral_2_eig, sampled_spectral, spectral, sampled_spectral_invpm


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
spectral_invpm(G)

print(time.time()-start)
