import networkx as nx
import numpy as np
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
import math


# PageRank Update Rule
def pur(G, nodes, rank, s):
    new_rank = {}

    for node in nodes:
        new_rank[node] = (1-s)/G.number_of_nodes()
        for x in G.neighbors(node):
            new_rank[node] += s*rank[x]/G.degree(x)

    return new_rank


# PageRank algorithm implemented in his dict version with k repetition of update rule and scaling factor s
def dict_pagerank(G, k, s):
    nodes = G.nodes()
    n = G.number_of_nodes()
    rank = {}

    # Set the initials rank values
    for node in nodes:
        rank[node] = 1/n

    # k update rule
    for i in range(k):
        rank = pur(G, nodes, rank, s)

    # Add all the node in a PQ and save the top 500
    pq_rank = PriorityQueue()
    for node in nodes:
        pq_rank.add(node, -rank[node])

    i = 0
    with open("PAGERANK/dict_pagerank/rank" + str(s) + ".txt", "w") as f1:
        while i < 500:
            node = pq_rank.pop()
            f1.write(node + '\n')
            i += 1


# PageRank algorithm implemented in his matrix version with k repetition of update rule and scaling factor s
def matrix_pagerank(G, k, s):
    n = G.number_of_nodes()

    # Dizionario che associa ogni nodo al suo rispettivo indice nella matrice
    nodes = {}
    i = 0
    for x in G.nodes():
        nodes[x] = i
        i += 1
    # Rank vector initial set-up
    rank = np.ones(n)/n

    # Adjacency matrix of the stochastic graph
    M = nx.adjacency_matrix(nx.stochastic_graph(G.to_directed()), G.nodes())

    # k-times update
    for i in range(k):
        # We make the product and the we scale the resulting rank
        rank = M.T.dot(rank)
        rank = s*rank
        for j in range(len(rank)):
            rank[j] += (1-s)/n

    pq_rank = PriorityQueue()
    for i in nodes.keys():
        pq_rank.add(i, -rank[nodes[i]])

    i = 0
    with open("PAGERANK/matrix_pagerank/rank" + str(s) + ".txt", "w") as f1:
        while i < 500:
            node = pq_rank.pop()
            f1.write(node + '\n')
            i += 1


# Parallel versions


# Utility used for split a vector data in chunks of the given size.
# Function used by the parallel implementation
def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in it.islice(idata, size)}


# Shared dictionary used bt parallel version of dict pagerank
shared_rank = {}


# PageRank Update Rule
def p_dict_pagerank(G, nodes, k, s):
    # We put all the node because a not perfect synchronisation can give at the next step
    # a not complete data structure
    for node in G.nodes:
        shared_rank[node] = 1/G.number_of_nodes()

    # k update rule
    for i in range(k):
        rank = shared_rank.copy()

        rank = pur(G, nodes, rank, s)

        shared_rank.update(rank)

    return shared_rank


# PageRank algorithm implemented in his dict version with k repetition of update rule and scaling factor s
def parallel_dict_pagerank(G, k, s, j):
    nodes = G.nodes()

    # Jobs require memory sharing
    with Parallel(n_jobs=j, require='sharedmem') as parallel:
        results = parallel(delayed(p_dict_pagerank)(G, X, k, s) for X in chunks(nodes, math.ceil(len(nodes) / j)))

    rank = {x: 0 for x in G.nodes()}
    for res in results:
        for x in res.keys():
            rank[x] += res[x]

    rank = {x: y/j for x, y in rank.items()}

    # Add all the node in a PQ and save the top 500
    pq_rank = PriorityQueue()
    for node in nodes:
        pq_rank.add(node, -rank[node])

    i = 0
    with open("PAGERANK/parallel_dict_pagerank/rank" + str(j) + ".txt", "w") as f1:
        while i < 500:
            node = pq_rank.pop()
            f1.write(node + '\n')
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
def p_matrix_pagerank(M, rank, n, k, s):
    # We put rank in shared_vec to initialize it
    shared_vec = rank.copy()

    # k update rule. We make M.dot(rank) because we chunk M.T instead M
    for i in range(k):
        rank = shared_vec.copy()
        
        x = list(M.keys())[0]
        rank = M[x].dot(rank)
        rank = s * rank
        for j in range(len(rank)):
            rank[j] += (1 - s) / n

        # x, which is the key of M in this thread, is also the starting vector index to be
        # update by this threads
        shared_vec[x:x+len(rank)] = rank

    return shared_vec


# Parallel function of the matrix version
def parallel_matrix_pagerank(G, k, s, j):
    n = G.number_of_nodes()

    # Dizionario che associa ogni nodo al suo rispettivo indice nella matrice
    nodes = {}
    i = 0
    for x in G.nodes():
        nodes[x] = i
        i += 1

    rank = np.ones(n) / n
    M = nx.adjacency_matrix(nx.stochastic_graph(G.to_directed()), G.nodes())

    # Threads require shared memory
    with Parallel(n_jobs=j, require='sharedmem') as parallel:
        # We chunk the transposed matrix, so in the update we make M*v product instead of M.T*v
        results = parallel(delayed(p_matrix_pagerank)(X, rank, n, k, s)
                          for X in chunks_matrix(M.T, math.ceil(len(rank) / j)))

    rank = np.zeros(n)
    for res in results:
        for i in range(len(res)):
            rank[i] += res[i]
    rank /= 4

    pq_rank = PriorityQueue()
    for i in nodes.keys():
        pq_rank.add(i, -results[0][nodes[i]])

    i = 0
    with open("PAGERANK/parallel_matrix_pagerank/rank" + str(j) + ".txt", "w") as f1:
        while i < 500:
            node = pq_rank.pop()
            f1.write(node + '\n')
            i += 1


