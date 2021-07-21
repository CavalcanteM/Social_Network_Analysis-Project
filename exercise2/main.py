import csv
import networkx as nx
import time
from hits import dict_hits, matrix_hits, parallel_matrix_hits, parallel_dict_hits
from pagerank import dict_pagerank, matrix_pagerank, parallel_dict_pagerank, parallel_matrix_pagerank
from degree import degree_centrality, parallel_degree_centrality
from closeness import closeness_centrality, parallel_closeness_centrality, sampled_closeness_centrality
from betweenness import betweenness_centrality, parallel_betweenness_centrality, sampled_betweenness_centrality


def load_graph():

    G = nx.Graph()

    with open('../facebook_large/musae_facebook_edges.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G


G = load_graph()

print("SAMPLED VERSION")
start = time.time()
sampled_betweenness_centrality(G, 4, 0.05)
print(time.time()-start)
