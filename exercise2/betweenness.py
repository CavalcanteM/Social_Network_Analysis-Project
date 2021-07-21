import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
import math
import random

# Computes vertex betweenness of the graph in input
def betweenness_centrality(G):
    node_btw = {i: 0 for i in G.nodes()}

    for s in G.nodes():
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in G.nodes()}  # the number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e): 0 for e in G.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in G.nodes()}  # The number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

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
                    distance[i] = distance[c]+1
                if distance[i] == distance[c]+1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while len(tree) > 0:
            c = tree.pop()
            for i in parents[c]:
                eflow[frozenset({c, i})] += vflow[c] * (spnum[i]/spnum[c])  #  the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[frozenset({c, i})]  #  each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                #  edge_btw[frozenset({c, i})] += eflow[frozenset({c,i})] #betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c] += vflow[c]  #  betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    # Save the top500
    pq = PriorityQueue()
    for node in G.nodes():
        pq.add(node, -node_btw[node])

    i = 0
    with open("BETWEENNESS/betweenness.txt", "w") as f:
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


# Function used by the parallel version
def parallel_betweenness(G, nodes):
    node_btw = {i: 0 for i in G.nodes()}

    for s in nodes:
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        distance = {i: -1 for i in G.nodes()}  # the number of shortest paths starting from s that use the edge e
        eflow = {frozenset(e): 0 for e in G.edges()}  # the number of shortest paths starting from s that use the edge e
        vflow = {i: 1 for i in G.nodes()}  # The number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

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
                    distance[i] = distance[c]+1
                if distance[i] == distance[c]+1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while len(tree) > 0:
            c = tree.pop()
            for i in parents[c]:
                eflow[frozenset({c, i})] += vflow[c] * (spnum[i]/spnum[c])  #  the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[frozenset({c, i})]  #  each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                #  edge_btw[frozenset({c, i})] += eflow[frozenset({c,i})] #betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c] += vflow[c]  #  betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return node_btw


# Computes edge and vertex betweenness of the graph in input
def parallel_betweenness_centrality(G, j):
    # parallel call with j job
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(parallel_betweenness)(G, X)
                          for X in chunks(G.nodes(), math.ceil(G.number_of_nodes() / j)))

    # Result fusion of the different job
    node_btw = {node: 0 for node in G.nodes()}
    for res in result:
        for node in res.keys():
            node_btw[node] += res[node]

    # Add all the node in a PQ and save the top 500
    pq = PriorityQueue()
    for node in G.nodes():
        pq.add(node, -node_btw[node])

    i = 0
    with open("BETWEENNESS/parallel_betweenness" + str(j) + ".txt", "w") as f:
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


# Computes vertex betweenness of the graph in input, but taking into account the BFS on a subset of nodes.
# In this version we use the parallel implementation because in the Naive version we prove that he return exactly
# same result, but in a faster way.
def sampled_betweenness_centrality(G, j, ratio):
    # Samplig operation
    nodes = random.sample(G.nodes(), int(len(G.nodes()) * ratio))
    print(len(nodes))
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(parallel_betweenness)(G, X)
                          for X in chunks_set(nodes, math.ceil(len(nodes) / j)))

    # Result fusion of the different job
    node_btw = {node: 0 for node in G.nodes()}
    for res in result:
        for node in res.keys():
            node_btw[node] += res[node]

    # Add all the node in a PQ and save the top 500
    pq = PriorityQueue()
    for node in G.nodes():
        pq.add(node, -node_btw[node])

    i = 0
    with open("BETWEENNESS/sampled" + str(ratio) + "_betweenness" + str(j) + ".txt", "w") as f:
        while i < 500:
            node = pq.pop()
            f.write(node + '\n')
            i += 1
