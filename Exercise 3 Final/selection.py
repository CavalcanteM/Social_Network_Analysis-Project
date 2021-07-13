import networkx as nx
from priorityq import PriorityQueue
import math, random
import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
import numpy as np


# Computes edge and vertex betweenness of the graph in input
def betweenness(G, X):
    edge_btw = {frozenset(e): 0 for e in G.edges()}
    for s in X:
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in G.nodes()}  # the number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e): 0 for e in G.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in G.nodes()}  # the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        # BFS
        queue = [s]
        spnum[s] = 1
        distance[s] = 0
        while len(queue) > 0:
            c = queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1:  # if vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c] + 1
                if distance[i] == distance[c] + 1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while len(tree) > 0:
            c = tree.pop()
            for i in parents[c]:
                eflow[frozenset({c, i})] += vflow[c] * (spnum[i] / spnum[c])  # the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[frozenset({c, i})]  # each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c, i})] += eflow[frozenset({c, i})]  # betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge

    return edge_btw


# Utility used for split a vector data in chunks of the given size.
# Function used by the parallel implementation
def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k for k in it.islice(idata, size)}


# Clusters are computed by iteratively removing edges of largest betweenness
def bwt_cluster(G, j):
    sampled = random.sample(G.nodes(), int(len(G.nodes()) * 0.05))

    with Parallel(n_jobs=j) as parallel:
        results = parallel(delayed(betweenness)(G, X) for X in chunks(sampled, math.ceil(len(sampled) / j)))

    edge_btw = {frozenset(e): 0 for e in G.edges()}
    for res in results:
        for edge in res.keys():
            edge_btw[edge] += res[edge]

    pq = PriorityQueue()
    for edge in edge_btw:
        pq.add(edge, -edge_btw[edge])

    done = False
    while not done:
        edge = tuple(sorted(pq.pop()))
        G.remove_edges_from([edge])
        clusters = list(nx.connected_components(G))

        if len(clusters) >= 4:
            done = True

    graphs = []
    for cluster in clusters:
        graphs.append(nx.subgraph(G, cluster))

    return graphs




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

    return pq


# HITS algorithm implemented in his matrix version
def matrix_hits(G):
    auth_hub = np.ones(G.number_of_nodes())
    nodes = sorted(G.nodes())
    M = nx.adjacency_matrix(G, nodes)

    done = 0
    # k-times product between the matrix and the vector
    while done == G.number_of_nodes():
        done = 0
        new_auth_hub = M.dot(auth_hub)

        auth_hub_norm = np.linalg.norm(auth_hub)

        new_auth_hub = new_auth_hub / auth_hub_norm

        for i in range(len(new_auth_hub)):
            if abs(auth_hub[i] - new_auth_hub[i]) < 10 ** -14:
                done += 1

        auth_hub = new_auth_hub.copy()

    pq_auth_hub = PriorityQueue()
    for i in range(len(nodes)):
        pq_auth_hub.add(nodes[i], -auth_hub[i])

    return pq_auth_hub

# PageRank algorithm implemented in his matrix version with k repetition of update rule and scaling factor s
def matrix_pagerank(G, s):
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
    done = 0
    # k-times update
    while done == G.number_of_nodes():
        done = 0
        # We make the product and the we scale the resulting rank
        new_rank = M.T.dot(rank)
        new_rank = s*new_rank
        for j in range(len(new_rank)):
            new_rank[j] += (1-s)/n

            if abs(new_rank[j] - rank[j]) < 10**-14:
                done += 1
        rank = new_rank.copy()

    pq_rank = PriorityQueue()
    for i in nodes.keys():
        pq_rank.add(i, -rank[nodes[i]])

    return pq_rank


def selector(G, B):
    # Calcolo clusters
    graphs = bwt_cluster(G, 4)

    pqs1 = []
    pqs2 = []
    pqs3 = []
    max = 0
    max_i = -1
    i = 0
    el = []
    print("NUMERO DI NODI NEI CLUSTER")
    for graph in graphs:
        # Verifica del cluster più numeroso
        if max < graph.number_of_nodes():
            max = graph.number_of_nodes()
            max_i = i
        print(graph.number_of_nodes())
        # Salvataggio dei seed da selezionare per ogni cluster
        el.append(int(B*graph.number_of_nodes()/G.number_of_nodes()))
        if el[-1] > 0:
            pqs1.append(matrix_hits(graph))
            pqs2.append(matrix_pagerank(graph, 0.85))
            pqs3.append(degree_centrality(graph))
        else:
            pqs1.append(PriorityQueue())
            pqs2.append(PriorityQueue())
            pqs3.append(PriorityQueue())
        i = i + 1

    # aggiungiamo i restantanti elementi (rimasti a causa del troncamento) al cluster più numeroso
    el[max_i] += B - el[0] - el[1] - el[2] - el[3]

    print(el[0]+el[1]+el[2]+el[3])

    # selezioniamo i seed tra le varie Priority Queues
    seeds1 = []
    seeds2 = []
    seeds3 = []
    for i in range(len(graphs)):
        for j in range(el[i]):
            seeds1.append(int(pqs1[i].pop()))
            seeds2.append(int(pqs2[i].pop()))
            seeds3.append(int(pqs3[i].pop()))
    return [seeds1, seeds2, seeds3]
