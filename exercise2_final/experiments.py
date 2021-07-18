import csv
import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools as it
import math
from joblib import Parallel, delayed
from lesson1 import stream_diam, num_triangles, diameter
from exercise2.betweenness import chunks_set, parallel_betweenness
from exercise2.closeness import chunks, parallel_closeness
from lesson4 import GenWS2DG
import time

def load_graph():

    G = nx.Graph()

    with open('net_13', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G

def degree(G, nodes):
    cen = dict()
    for u in nodes:
        cen[u] = G.degree(u)
    return cen

def sampled_betweenness_centrality(G, j):
    
    nodes = random.sample(G.nodes(), int(len(G.nodes()) * 0.1))
    
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(parallel_betweenness)(G, X)
                          for X in chunks_set(nodes, math.ceil(len(nodes) / j)))

    node_btw = {node: 0 for node in G.nodes()}
    for res in result:
        for node in res.keys():
            node_btw[node] += res[node]

    return node_btw

def parallel_closeness_centrality(G, j):
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(parallel_closeness)(G, X)
                          for X in chunks(G.nodes(), math.ceil(G.number_of_nodes() / j)))
        
    agg_result = dict()
    for cen in result:
      for node in cen.keys():
        agg_result[node] = cen[node]
    
    return agg_result

G = load_graph()
# compute diameter
diameter = stream_diam(G)
# compute number of triangles
triangles = num_triangles(G)
# compute degree distribution
dd = degree(G, G.nodes())
# compute node betweenness
node_btw = sampled_betweenness_centrality(G, 4)
# compute node closeness
closeness = parallel_closeness_centrality(G, 4)
# compute clustering coefficient
sum = 0
for k in G.nodes():
  sum += G.degree(k)*(G.degree(k)-1)

clustering_coeff = (3 * triangles) / sum

# r = 3
# k = 100
# k = 70
# k = 40
# k = 30
# q = 1.5

# Watts-Strongatz parameters
n = 10000
r = 5
k = 10
q = 2

# Generate proposed network
start = int(time.time())
G1 = GenWS2DG(n, r, k, q)
end = int(time.time())
print("Rete generata - Tempo impiegato: ", int((end - start)/60), "minuti")

print("Diametro rete originale: ", diameter)
print("Diametro rete generata: ", stream_diam(G1))

print("Triangoli rete originale: ", triangles)
triangles1 = num_triangles(G1)
print("Triangoli rete generata: ", triangles1)

# Plot original degree distribution
plt.hist(dd.values())
plt.title("Original Degree distribution")
plt.xlabel("Value")
plt.ylabel("Number of nodes")
plt.show()

dd1 = degree(G1, G1.nodes())
# Plot generated degree distribution
plt.hist(dd1.values())
plt.title("Generated Degree distribution")
plt.xlabel("Value")
plt.ylabel("Number of nodes")
plt.show()

# Plot original node betweenness distribution
plt.hist(node_btw.values())
plt.title("Original node btw distribution")
plt.xlabel("Value")
plt.ylabel("Number of nodes")
plt.show()

node_btw1 = sampled_betweenness_centrality(G1, 4)
# Plot generated node betweenness distribution
plt.hist(node_btw1.values())
plt.title("Generated node btw distribution")
plt.xlabel("Value")
plt.ylabel("Number of nodes")
plt.show()

# Plot node betweenness distribution
plt.hist(closeness.values())
plt.title("Original closeness distribution")
plt.xlabel("Value")
plt.ylabel("Number of nodes")
plt.show()

closeness1 = parallel_closeness_centrality(G1, 4)
# Plot node betweenness distribution
plt.hist(closeness1.values())
plt.title("Generated closeness distribution")
plt.xlabel("Value")
plt.ylabel("Number of nodes")
plt.show()

print("Clustering coefficient rete originale: ", clustering_coeff)

sum = 0
for k in G1.nodes():
  sum += G1.degree(k)*(G1.degree(k)-1)

clustering_coeff1 = (3 * triangles1) / sum
print("Clustering coefficient rete generata: ", clustering_coeff1)