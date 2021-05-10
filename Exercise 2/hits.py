import networkx as nx
import numpy as np
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
import math


# Update Rule
def aur_hur(G, nodes, X):
    new = {}
    for node in nodes:
        new[node] = 0
        for x in G.neighbors(node):
            new[node] = new[node] + X[x]

    return new


# normalization of the value of a dictionary given their norm
def normalization(d, norm):
    for x in d.keys():
        d[x] = d[x] / norm
    return d


# Hits implementation using dictionary
def dict_hits(G, n):
    nodes = sorted(G.nodes())
    auth_hub = {}

    # At the start all the nodes have value equal to 1
    for node in nodes:
        auth_hub[node] = 1

    # n execution of the Update Rule, with normalization in each step
    for k in range(n):
        auth_hub = aur_hur(G, nodes, auth_hub)

        auth_hub_norm = np.linalg.norm(list(auth_hub.values()))

        auth_hub = normalization(auth_hub, auth_hub_norm)

    # Add all the node in a PQ and save the top 500
    pq_auth_hub = PriorityQueue()
    for node in nodes:
        pq_auth_hub.add(node, -auth_hub[node])

    i = 0
    with open("HITS/dict_hits/hits.txt", "w") as f1:
        while i < 500:
            f1.write(pq_auth_hub.pop() + '\n')
            i += 1


# HITS algorithm implemented in his matrix version
def matrix_hits(G, n):
    auth_hub = np.ones(G.number_of_nodes())
    nodes = sorted(G.nodes())
    M = nx.adjacency_matrix(G, nodes)

    # k-times product between the matrix and the vector
    for k in range(n):
        auth_hub = M.dot(auth_hub)

        auth_hub_norm = np.linalg.norm(auth_hub)

        auth_hub = auth_hub / auth_hub_norm

    pq_auth_hub = PriorityQueue()
    for i in range(len(nodes)):
        pq_auth_hub.add(nodes[i], -auth_hub[i])

    i = 0
    with open("HITS/matrix_hits/hits.txt", "w") as f1:
        while i < 500:
            f1.write(pq_auth_hub.pop() + '\n')
            i += 1


# Utility used for split a vector data in chunks of the given size.
# Function used by the parallel implementation
def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in it.islice(idata, size)}


# Shared dict used by threads of dict version of hits
shared_auth_hub = {}


# Function that run the k Update Rule in parallel.
# It uses the shared_auth_hub for share the updates among threads
def p_dict_hits(G, nodes, n):
    # We put all the node because a not perfect synchronisation can give at the next step
    # a not complete data structure
    for node in G.nodes():
        shared_auth_hub[node] = 1

    for k in range(n):
        # We copy the shared dict to reduce lock time
        auth_hub = shared_auth_hub.copy()

        # Before to do the Update Rule, we normalize the values of the dict.
        # We do it before and no after because if we make it after we have no proof that the values
        # are normalized (after the normalization in this thread, another thread can start to update it).
        # Moreover in this way we can update shared_auth_hub only once per cycle instead twice.
        auth_hub_norm = np.linalg.norm(list(auth_hub.values()))
        auth_hub = normalization(auth_hub, auth_hub_norm)

        # UR
        auth_hub = aur_hur(G, nodes, auth_hub)

        # Update the shared dict
        shared_auth_hub.update(auth_hub)

    return shared_auth_hub


# Parallel function of the dictionary version
def parallel_dict_hits(G, n, j):
    nodes = G.nodes()

    # Jobs require memory sharing
    with Parallel(n_jobs=j, require='sharedmem') as parallel:
        results = parallel(delayed(p_dict_hits)(G, X, n) for X in chunks(nodes, math.ceil(len(nodes) / j)))

        # Normalization of the results
        auth_hub_norm = np.linalg.norm(list(results[0].values()))

        auth_hub = normalization(results[0], auth_hub_norm)

    pq_auth_hub = PriorityQueue()
    for node in nodes:
        pq_auth_hub.add(node, -auth_hub[node])

    i = 0
    with open("HITS/parallel_dict_hits/hits.txt", "w") as f1:
        while i < 500:
            f1.write(pq_auth_hub.pop() + '\n')
            i += 1


# Utility used for split a matrix in chunks of the given size.
# The returned value is a dictionary in which the key represents the starting index of the chunks.
# Function used by the parallel implementation
def chunks_matrix(data, size):
    for i in range(0, data.shape[0], size):
        yield {i: data[i:i+size]}


# Shared vector used by threads of the matrix version of hits
shared_vec = []


# Function that run the k Update Rule in parallel.
# It uses the shared_vec for share the updates among threads
def p_matrix_hits(M, nodes, n):
    # We put len(nodes) 1 in shared_vec
    for i in range(len(nodes)):
        if len(shared_vec) < len(nodes):
            shared_vec.append(1)
        else:
            break

    # Execution of n UR
    for k in range(n):
        auth_hub = shared_vec.copy()

        # Before to do the Update Rule, we normalize the vector.
        # We do it before and no after because if we make it after we have no proof that the vector
        # is normalized (after the normalization in this thread, another thread can start to update it).
        # Moreover in this way we can update shared_vec only once per cycle instead twice.
        auth_hub_norm = np.linalg.norm(auth_hub)
        auth_hub = auth_hub / auth_hub_norm

        # UR rule
        j = list(M.keys())[0]
        new_auth_hub = M[j].dot(auth_hub)

        # j, which is the key of M in this thread, is also the starting vector index to be
        # update by this threads
        for i in range(len(new_auth_hub)):
            shared_vec[j + i] = new_auth_hub[i]

    return shared_vec


# Parallel function of the matrix version
def parallel_matrix_hits(G, n, j):
    nodes = sorted(G.nodes())
    M = nx.adjacency_matrix(G, nodes)

    # Threads require shared memory
    with Parallel(n_jobs=j, require='sharedmem') as parallel:
        results = parallel(delayed(p_matrix_hits)(X, nodes, n)
                          for X in chunks_matrix(M, math.ceil(len(nodes) / j)))

    # We normalize the result. All the j result among threads are equal.
    auth_hub_norm = np.linalg.norm(results[0])
    auth_hub = results[0] / auth_hub_norm

    pq_auth_hub = PriorityQueue()
    for i in range(len(nodes)):
        pq_auth_hub.add(nodes[i], -auth_hub[i])

    i = 0
    with open("HITS/parallel_matrix_hits/hits.txt", "w") as f1:
        while i < 500:
            f1.write(pq_auth_hub.pop() + '\n')
            i += 1
