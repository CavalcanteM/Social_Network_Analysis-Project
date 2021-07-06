import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
import math


# The measure associated to each node is exactly its degree divided by number of nodes - 1
def degree(G, nodes):
    cen = dict()
    for u in nodes:
        cen[u] = G.degree(u) / (G.number_of_nodes() - 1)
    return cen


# Naive implementation of degree centrality
def degree_centrality(G):
    cen = degree(G, G.nodes())

    # Add all the node in a PQ and save the top 500
    pq = PriorityQueue()
    for node in cen.keys():
        pq.add(node, -cen[node])

    i = 0
    with open("DEGREE/degree.txt", "w") as f:
        while i < 500:
            node = pq.pop()
            f.write(node + '\n')
            i += 1


# Utility used for split a vector data in chunks of the given size.
# Function used by the parallel implementation
def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in it.islice(idata, size)}


# Parallel implementation of degree centrality
def parallel_degree_centrality(G, j):
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(degree)(G, X)
                          for X in chunks(G.nodes(), math.ceil(G.number_of_nodes() / j)))


    # Add all the node in a PQ and save the top 500
    pq = PriorityQueue()
    for cen in result:
        for node in cen.keys():
            pq.add(node, -cen[node])

    i = 0
    with open("DEGREE/parallel_degree" + str(j) + ".txt", "w") as f:
        while i < 500:
            node = pq.pop()
            f.write(node + '\n')
            i += 1