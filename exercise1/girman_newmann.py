import math, random
import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
from functions import save_clusters


# Computes edge betweenness of the graph in input starting from the nodes passes in input
# Used from all the version of this algorithm
def betweenness(G,X):
    edge_btw={frozenset(e):0 for e in G.edges()}
    for s in X:
        # Compute the number of shortest paths from s to every other node
        tree=[] #it lists the nodes in the order in which they are visited
        spnum={i:0 for i in G.nodes()} #it saves the number of shortest paths from s to i
        parents={i:[] for i in G.nodes()} #it saves the parents of i in each of the shortest paths from s to i
        distance={i:-1 for i in G.nodes()} #the number of shortest paths starting from s that use the edge e
        eflow={frozenset(e):0 for e in G.edges()} #the number of shortest paths starting from s that use the edge e
        vflow={i:1 for i in G.nodes()} #the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        #BFS
        queue=[s]
        spnum[s]=1
        distance[s]=0
        while queue != []:
            c=queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1: #if vertex i has not been visited
                    queue.append(i)
                    distance[i]=distance[c]+1
                if distance[i] == distance[c]+1: #if we have just found another shortest path from s to i
                    spnum[i]+=spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c=tree.pop()
            for i in parents[c]:
                eflow[frozenset({c,i})]+=vflow[c] * (spnum[i]/spnum[c]) #the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i]+=eflow[frozenset({c,i})] #each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c,i})]+=eflow[frozenset({c,i})] #betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge

    return edge_btw


# Naive implementation of betweenness clustering
# Clusters are computed by iteratively removing edges of largest betweenness
def bwt_cluster_naive(G):
    edge_btw = betweenness(G, G.nodes())

    pq = PriorityQueue()
    for i in edge_btw.keys():
        pq.add(i, -edge_btw[i])

    # Stop the edge removing when we obtain 4 clusters
    clusters = list()
    while len(clusters) < 4:
        edge = tuple(sorted(pq.pop()))
        G.remove_edges_from([edge])
        clusters = list(nx.connected_components(G))

    # We save each cluster in a different file
    save_clusters("GIRMAN_NEWMANN/naive", clusters[0], clusters[1], clusters[2], clusters[3])


def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k for k in it.islice(idata, size)}


# Parallel implementation with j jobs
def bwt_cluster_parallel(G, j):
    with Parallel(n_jobs=j) as parallel:
        results = parallel(delayed(betweenness)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

    edge_btw = {frozenset(e): 0 for e in G.edges()}

    for res in results:
        for edge in res.keys():
            edge_btw[edge] += res[edge]

    pq = PriorityQueue()
    for i in edge_btw.keys():
        pq.add(i, -edge_btw[i])

    clusters = list()
    while len(clusters) < 4:
        edge = tuple(sorted(pq.pop()))
        G.remove_edges_from([edge])
        clusters = list(nx.connected_components(G))

    # We save each cluster in a different file
    save_clusters("GIRMAN_NEWMANN/parallel", clusters[0], clusters[1], clusters[2], clusters[3])


# Sampled implementation using a fraction of node given by ratio. It is implemented in parallel mode with j jobs
def bwt_cluster_sampled(G, ratio, j):
    sampled = random.sample(G.nodes(), int(len(G.nodes())*ratio))
    with Parallel(n_jobs=j) as parallel:
        results = parallel(delayed(betweenness)(G, X) for X in chunks(sampled, math.ceil(len(sampled) / j)))

    edge_btw = {frozenset(e): 0 for e in G.edges()}

    for res in results:
        for edge in res.keys():
            edge_btw[edge] += res[edge]

    pq = PriorityQueue()
    for i in edge_btw.keys():
        pq.add(i, -edge_btw[i])

    clusters = list()
    while len(clusters) < 4:
        edge = tuple(sorted(pq.pop()))
        G.remove_edges_from([edge])
        clusters = list(nx.connected_components(G))

    # We save each cluster in a different file
    save_clusters("GIRMAN_NEWMANN/sampled", clusters[0], clusters[1], clusters[2], clusters[3])