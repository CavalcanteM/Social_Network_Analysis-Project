import math, random
import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it


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
    with open("GIRMAN_NEWMANN/naive/cluster0.txt", "w") as f:
        for element in clusters[0]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/naive/cluster1.txt", "w") as f:
        for element in clusters[1]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/naive/cluster2.txt", "w") as f:
        for element in clusters[2]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/naive/cluster3.txt", "w") as f:
        for element in clusters[3]:
            f.write(element + "\n")


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
    with open("GIRMAN_NEWMANN/parallel/cluster0.txt", "w") as f:
        for element in clusters[0]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/parallel/cluster1.txt", "w") as f:
        for element in clusters[1]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/parallel/cluster2.txt", "w") as f:
        for element in clusters[2]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/parallel/cluster3.txt", "w") as f:
        for element in clusters[3]:
            f.write(element + "\n")


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
    with open("GIRMAN_NEWMANN/sampled/cluster0.txt", "w") as f:
        for element in clusters[0]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/sampled/cluster1.txt", "w") as f:
        for element in clusters[1]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/sampled/cluster2.txt", "w") as f:
        for element in clusters[2]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/sampled/cluster3.txt", "w") as f:
        for element in clusters[3]:
            f.write(element + "\n")


# Counts the edges that connect nodes inside the same cluster
def intra_community(G, clusters):
    value = 0
    for edge in G.edges():
        for cluster in clusters:
            if edge[0] in cluster and edge[1] in cluster:
                value +=1
                break
    return value


# Counts the edges that doesn't connect nodes from different clusters
def inter_community_non_edges(G, clusters):
    value = 0
    for i in range(len(clusters)):
        for node1 in clusters[i]:
            for j in range(i+1,len(clusters)):
                for node2 in clusters[j]:
                    if node2 not in G.neighbors(node1):
                        value += 1
    return value


# Evaluates a performance of the proposed partitions [0,1]
def performance(G, partitions):
    intra_community_n = intra_community(G, partitions)
    inter_community_non_edges_n = inter_community_non_edges(G, partitions)
    n = len(G.nodes())
    total_pairs = n * (n - 1)
    print("Valori performance", intra_community_n, inter_community_non_edges_n, total_pairs)
    return (intra_community_n + inter_community_non_edges_n) / total_pairs


# To try to get an highest accuracy we remove edge until the performance value is > 0
def bwt_cluster_performance_based(G, ratio, j):
    sampled = random.sample(G.nodes(), int(len(G.nodes())*ratio))
    print("NODI CAMPIONATI:", len(sampled))
    with Parallel(n_jobs=j) as parallel:
        results = parallel(delayed(betweenness)(G,X) for X in chunks(sampled, math.ceil(len(sampled)/j)))
    
    edge_btw = {frozenset(e): 0 for e in G.edges()}

    for res in results:
        for edge in res.keys():
            edge_btw[edge] += res[edge]

    pq=PriorityQueue()

    for edge in edge_btw:
        pq.add(edge, -edge_btw[edge])

    graph=G.copy()

    current_clusters = list(nx.connected_components(graph))
    lprec = len(current_clusters)
    perf = performance(G,current_clusters)
    
    # remove the highest edge until the performance value reaches 0.5
    while perf < 0.5:
        highest_edge = tuple(sorted(pq.pop()))
        graph.remove_edges_from([highest_edge])
        current_clusters = list(nx.connected_components(graph))
        if len(current_clusters) != lprec:
            lprec = len(current_clusters)
            perf = performance(G,current_clusters)

    current_clusters.sort(key=len, reverse=True)

    # We save each cluster in a different file
    with open("GIRMAN_NEWMANN/performance_based/cluster0.txt", "w") as f:
        for element in current_clusters[0]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/performance_based/cluster1.txt", "w") as f:
        for element in current_clusters[1]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/performance_based/cluster2.txt", "w") as f:
        for element in current_clusters[2]:
            f.write(element + "\n")

    with open("GIRMAN_NEWMANN/performance_based/cluster3.txt", "w") as f:
        for element in current_clusters[3]:
            f.write(element + "\n")
