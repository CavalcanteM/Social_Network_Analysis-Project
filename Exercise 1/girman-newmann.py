import os, csv, math, random
import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it


def likelihood(G,cluster1, cluster2):
    value = 0
    for edge in G.edges():
        if (edge[0] in cluster1 and edge[1] in cluster2) or (edge[0] in cluster2 and edge[1] in cluster1):
            value += 1
    return value/(len(cluster1)+len(cluster2))

# Counts the edges that connect nodes inside the same cluster
def intra_community(G,clusters):
    value = 0
    for edge in G.edges():
        for cluster in clusters:
            if edge[0] in cluster and edge[1] in cluster:
                value +=1
                break
    return value

# Counts the edges that doesn't connect nodes from different clusters
def inter_community_non_edges(G,clusters):
    value = 0
    for i in range(len(clusters)):
        for node1 in clusters[i]:
            for j in range(i+1,len(clusters)):
                for node2 in clusters[j]:
                    if node2 not in G.neighbors(node1):
                        value += 1
    return value

# Evaluates a performance of the proposed partitions [0,1]
def performance(G,partitions):
    intra_community_n = intra_community(G,partitions)
    inter_community_non_edges_n = inter_community_non_edges(G,partitions)
    n = len(G.nodes())
    total_pairs = n * (n - 1)
    print("Valori performance",intra_community_n, inter_community_non_edges_n, total_pairs)
    return (intra_community_n + inter_community_non_edges_n) / total_pairs

# Computes edge and vertex betweenness of the graph in input
def betweenness(G,X):
    edge_btw={frozenset(e):0 for e in G.edges()}
    node_btw={i:0 for i in G.nodes()}
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
            if c != s:
                node_btw[c]+=vflow[c] #betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw

#Clusters are computed by iteratively removing edges of largest betweenness
def bwt_cluster_parallel(G,j):
    
    sampled = random.sample(G.nodes(), int(len(G.nodes())*0.1))
    print("NODI CAMPIONATI:", len(sampled))
    with Parallel(n_jobs=j) as parallel:
        results = parallel(delayed(betweenness)(G,X) for X in chunks(sampled, math.ceil(len(sampled)/j)))
    
    edge_btw = {frozenset(e): 0 for e in G.edges()}

    for res in results:
        for edge in res.keys():
            edge_btw[edge] += res[edge]
    
    #print("RIEMPIO PQ")
    pq=PriorityQueue()

    for edge in edge_btw:
        pq.add(edge, -edge_btw[edge])
    
    #print("FINITO")
    graph=G.copy()

    current_clusters = list(nx.connected_components(graph))
    lprec = len(current_clusters)
    perf = performance(G,current_clusters)
    #print("Performance",perf,len(current_clusters))

    #while len(current_clusters) < 4:
    #    highest_edge = tuple(sorted(pq.pop()))
    #    graph.remove_edges_from([highest_edge])
    #    current_clusters = list(nx.connected_components(graph))
    
    # remove the highest edge until the performance value reaches 0.5
    while perf < 0.5:
        highest_edge = tuple(sorted(pq.pop()))
        #print("RIMOSSO",highest_edge, len(graph.edges()))
        graph.remove_edges_from([highest_edge])
        current_clusters = list(nx.connected_components(graph))
        if len(current_clustersl) != lprec:
            lprec = len(current_clusters)
            p = performance(G,current_clusters)
            #print("Performance",p,len(current_clusters))

    #print(current_clusters)
    #print(len(current_clusters))

    # In case there are too many clusters, we fuse together the most connected couple of clusters
    while len(current_clusters) > 4:
        best_choice = PriorityQueue()

        for i in range(len(current_clusters)):
            for j in range (i+1, len(current_clustersl)):

                lik = likelihood (G, current_clusters[i], current_clusters[j])
                best_choice.add((i,j), -lik)
        
        (a,b) = best_choice.pop()
        ca = current_clusters.pop(a)
        cb = current_clusters.pop(b-1)
        current_clusters.append(ca+cb)

    return current_clusters


def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k for k in it.islice(idata, size)}

G = nx.Graph()

with open('facebook_large/musae_facebook_edges.csv', newline='') as csvfile:

    rows = csv.reader(csvfile, delimiter=',')
    next(rows)
    for row in rows:
        G.add_edge(row[0], row[1])

clusters = bwt_cluster_parallel(G,8)
