import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
import math
import random


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


# Naive implementation of closeness centrality
def closeness_centrality(G):
    pq = PriorityQueue()

    # Computation of closeness for each node
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


# Utility used for split a vector data in chunks of the given size.
# Function used by the parallel implementation
def chunks_set(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k for k in it.islice(idata, size)}


# Function used to select a subset of nodes in which the excluded elements are not neighbors to each other.
# We decided to include all the neighbors of an excluded node because for excluded nodes we compute their
# closeness as the mean of his neighbors' closeness
def sampling_operation(G):
    nodes = list(G.nodes())
    final_set = set()
    excluded = set()
    while len(nodes) > 0:
        # The random node chosen is not included in the final set, but his neighbors yes
        v = random.choice(nodes)
        nodes.remove(v)
        excluded.add(v)
        for u in G.neighbors(v):
            final_set.add(u)
            if u in nodes:
                nodes.remove(u)
    print(len(final_set))
    return final_set, excluded


# Sampled implementation of closeness centrality.
# For the excluded nodes, their closeness centrality is represented by the mean closeness centrality of their neighbors
def sampled_closeness_centrality(G, j):
    nodes, excluded = sampling_operation(G)
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(parallel_closeness)(G, X)
                          for X in chunks_set(nodes, math.ceil(len(nodes) / j)))

    # Union of the j dicts
    closeness = dict()
    for cen in result:
        closeness.update(cen)
    # Now, we calculate the closeness of the excluded node as the mean of the neighbors' closeness
    for v in excluded:
        sum = 0
        for u in G.neighbors(v):
            sum += closeness[u]
        closeness[v] = sum / G.degree(v)


    # Add all the node in a PQ and save the top 500
    pq = PriorityQueue()
    for node in closeness.keys():
        pq.add(node, -closeness[node])

    i = 0
    with open("CLOSENESS/sampled_closeness" + str(j) + ".txt", "w") as f:
        while i < 500:
            node = pq.pop()
            f.write(node + '\n')
            i += 1