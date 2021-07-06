import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
import math


# The closeness centrality is based on the distance between the node and all the other nodes.
# Because the graph is unweighted and undirected, we use BFS algorithm for this purpose.
# It return the sum of all the distance
def sum_bfs(G, u):
    visited = set()
    visited.add(u)
    queue = [u]
    dist = dict()
    dist[u] = 0
    sum = 0

    while len(queue) > 0:
        v = queue.pop(0)
        for w in G[v]:
            if w not in visited:
                visited.add(w)
                queue.append(w)
                dist[w] = dist[v]+1
                sum += dist[w]

    return sum

def closeness_centrality(G):
    pq = PriorityQueue()

    for node in G.nodes():
        closeness = (G.number_of_nodes()-1)/sum_bfs(G, node)
        pq.add(node, -closeness)

    i = 0
    with open("CLOSENESS/closeness.txt", "w") as f:
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


# Function used by the parallel version.
# It computes the closeness on the subset nodes.
def parallel_closeness(G, nodes):
    closeness = dict()
    for node in nodes:
        closeness[node] = (G.number_of_nodes()-1)/sum_bfs(G, node)
    return closeness


# Parallel implementation of closeness centrality
def parallel_closeness_centrality(G, j):
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(parallel_closeness)(G, X)
                          for X in chunks(G.nodes(), math.ceil(G.number_of_nodes() / j)))


    # Add all the node in a PQ and save the top 500
    pq = PriorityQueue()
    for cen in result:
        for node in cen.keys():
            pq.add(node, -cen[node])

    i = 0
    with open("CLOSENESS/parallel_closeness" + str(j) + ".txt", "w") as f:
        while i < 500:
            node = pq.pop()
            f.write(node + '\n')
            i += 1
