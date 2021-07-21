import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed
from priorityq import PriorityQueue
import random

#DIAMETER
#Classical algorithm: if runs a BFS for each node, and returns the height of the tallest BFS tree
#It is has computational complexity O(n*m)
#It require to keep in memory the full set of nodes (it may huge)
#It can be optimized by
#1) sampling only a subset of nodes on which the BFS is run (solution may be not exact, quality of the solution depends on the number of sampled nodes)
#2) parallelizing BFSs (depends on the available processing power)
#3) ad-hoc optimization
def diameter(G,sample=None):
    n = len(G.nodes())
    diam = 0
    if sample is None:
        sample = G.nodes()

    for u in sample:
        udiam=0
        clevel=[u]
        visited=set(u)
        while len(visited) < n:
            nlevel=[]
            while(len(clevel) > 0):
                c=clevel.pop()
                for v in G[c]:
                    if v not in visited:
                        visited.add(v)
                        nlevel.append(v)
            clevel = nlevel
            udiam += 1
        if udiam > diam:
            diam = udiam

    return diam

#PARALLEL IMPLEMENTATION
#Utility used for split a vector data in chunks of the given size.
def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

#Parallel implementation of diameter with joblib
def parallel_diam(G,j):
    diam = 0
    #Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        #Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        result=parallel(delayed(diameter)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results
        for res in result:
            if res > diam:
                diam = res
    return diam

#AD-HOC OPTIMIZATION
#This algorithm only returns an approximation of the diameter.
#It explores all edges multiple times for a number of steps that is approximately the diameter of the graph.
#Thus, it has computational complexity O(diam*m), that is usually much faster than the complexity of previous algorithm.
#
#The algorithm need to keep in memory a number for each node.
#There is a version of the algorithm that reduce the amount of used memory: It does not save the degree (that may be a large number), but uses hash functions and it is able to save few bits for each vertex.
#
#The main idea is to simulating a message broadcast by the node with maximum degree: the time it takes to reach all nodes it must be the diameter.
#To avoid to compute the node of maximum degree and simulate the above sending process, we have that at each step each node saves the maximum among its own degree and the degree of its neighbors.
#This approach may fail whenever there are many nodes broadcasting the same message.
def stream_diam(G):
    # At the beginning, R contains for each vertex v the number of nodes that can be reached from v in one step
    R={v:G.degree(v) for v in G.nodes()}

    step=0
    done=False
    while not done:
        step+=1
        done = True
        for edge in G.edges():
            # At the i-th iteration, we want that R contains for each vertex v an approximation of the number of nodes that can be reached from v in i+1 steps
            # If there is edge (u,v), then v can reach in i+1 steps at least the number of nodes that u can reach in i steps
            if R[edge[0]] != R[edge[1]]:
                R[edge[0]] = max(R[edge[0]],R[edge[1]])
                R[edge[1]] = max(R[edge[0]],R[edge[1]]) #This line must be removed in a directed graph
                done=False
    return step

# TRIANGLES
#Classical algorithm
#The problems of this algorithm are two:
#- The same triangle is counted multiple times (six times)
#- For each node, it requires to visit its neighborhood twice (and the neighborhood can be large).
#  For triangles involving only nodes of large degree this cost cannot be avoided even if we count the triangle only once (i.e., we address the first issue).
#  In other words these triangles are a bottleneck for the running time of this algorithm.
#  For the remaining triangles, if u is the node of smaller degree, then the time needed to find them is reasonable.
def triangles(G):
    triangles = 0

    for u in G.nodes():
        for v in G[u]:
            for w in G[u]:
                if w in G[v]:
                    triangles += 1

    return int(triangles / 6)

#Optimized algorithm
#There are two optimizations.
#
#OPTIMIZATION1: It consider an order among nodes. Specifically, nodes are ordered by degree. In case of nodes with the same degree, nodes are ordered by label.
#In this way a triangle is counted only once. Specifically, from the node with smaller degree to the one with larger degree.
def less(G, edge):
    if G.degree(edge[0]) < G.degree(edge[1]):
        return 0
    if G.degree(edge[0]) == G.degree(edge[1]) and edge[0] < edge[1]:
        return 0
    return 1

#OPTIMIZATION2: It distinguishes between high-degree nodes (called heavy hitters) and low-degree nodes.
#Triangles involving only heavy hitters (that have been recognized to be the bottleneck of the naive algorithm) are handled in a different way respect to remaining triangles.
def num_triangles(G):
    m=nx.number_of_edges(G)
    num_triangles = 0

    #The set of heavy hitters, that is nodes with degree at least sqrt(m)
    #Note: the set contains at most sqrt(m) nodes.
    #Note: the choice of threshold sqrt(m) is the one that minimize the running time of the algorithm.
    #A larger value of the threshold implies a faster processing of triangles containing only heavy hitters, but a slower processing of remaining triangles.
    #A smaller value of the threshold implies the reverse.
    heavy_hitters=set()
    for u in G.nodes():
        if G.degree(u) >= math.sqrt(m):
            heavy_hitters.add(u)

    #Number of triangles among heavy hitters.
    #It considers all possible triples of heavy hitters, and it verifies if it forms a triangle.
    #The running time is then O(sqrt(m)^3) = m*sqrt(m)
    for triple in it.combinations(heavy_hitters,3):
        if G.has_edge(triple[0],triple[1]) and G.has_edge(triple[0],triple[2]) and G.has_edge(triple[2],triple[1]):
            num_triangles += 1

    #Number of remaining triangles.
    #For each edge, if one of the endpoints is not an heavy hitter, verifies if there is a node in its neighborhood that forms a triangle with the other endpoint.
    #This is essentially the naive algorithm optimized to count only ordered triangle in which the first vertex (i.e., u) is not an heavy hitter.
    #Since the size of the neighborhood of a non heavy hitter is at most sqrt(m), the complexity is O(m*sqrt(m))
    for edge in G.edges(): #They are m
        sel=less(G,edge)
        if edge[sel] not in heavy_hitters: #If the endpoint of smaller degree is an heavy hitter, we skip this edge
            for i in G[edge[sel]]: #They are less than sqrt(m)
                if less(G,[i,edge[1-sel]]) and G.has_edge(i,edge[1-sel]): #In this way we count the triangle only once
                    num_triangles += 1

    return num_triangles

#STANDARD CLUSTERING ALGORITHMS
#They may not return significative results when run on network, since there is not a satisfying measure of distance
#Clustering in a graph does not depend only on the position of two neighbors, but also on the position of other nodes with respect to these two.
def hierarchical(G):
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 0)
                else:
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 1)

    # Start with a cluster for each node
    clusters = set(frozenset(u) for u in G.nodes())

    done = False
    while not done:
        # Merge closest clusters
        s = list(pq.pop())
        clusters.remove(s[0])
        clusters.remove(s[1])

        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset([s[0] | s[1], w]), 0)
            else:
                pq.add(frozenset([s[0] | s[1], w]), 1)

        clusters.add(s[0] | s[1])

        print(clusters)
        a = input("Do you want to continue? (y/n) ")
        if a == "n":
            done = True


def two_means(G):
    n=G.number_of_nodes()
    # Choose two clusters represented by vertices that are not neighbors
    u = random.choice(list(G.nodes()))
    v = random.choice(list(nx.non_neighbors(G, u)))
    cluster0 = {u}
    cluster1 = {v}
    added = 2

    while added < n:
        # Choose a node that is not yet in a cluster and add it to the closest cluster
        x = random.choice([el for el in G.nodes() if el not in cluster0|cluster1 and (len(
            set(G.neighbors(el)).intersection(cluster0)) != 0 or len(set(G.neighbors(el)).intersection(cluster1)) != 0)])
        if len(set(G.neighbors(x)).intersection(cluster0)) != 0:
            cluster0.add(x)
            added+=1
        elif len(set(G.neighbors(x)).intersection(cluster1)) != 0:
            cluster1.add(x)
            added+=1

    print(cluster0, cluster1)

# G=nx.Graph()
# G.add_edge('A', 'B')
# G.add_edge('A', 'C')
# G.add_edge('B', 'C')
# G.add_edge('B', 'D')
# G.add_edge('D', 'E')
# G.add_edge('D', 'F')
# G.add_edge('D', 'G')
# G.add_edge('E', 'F')
# G.add_edge('F', 'G')
# print(diameter(G))
# #Observe that on small graphs, as in this case, the overhead of dividing the input, aggregating the output and managing parallelization, makes the parallel algo less convenient than the non-parallel one.
# print(parallel_diam(G,2))
# print(stream_diam(G))
# print(triangles(G))
# print(num_triangles(G))
