import random
import networkx as nx
import time
from joblib import Parallel, delayed
import itertools as it
import math

#Initial phase in which we add one vertex to each cluster (we assume the presence of only 4 clusters).
#Each vertex is selected from a list of non neighbors of the before selected vertex (if and only if it is possible).
#If the list of non neighbors is empty we select a random and not already selected node from the graph.
def init_phase(G):
    start_init_phase = time.time()

    # Choose four clusters represented by vertices that are not neighbors
    u = random.choice(list(G.nodes()))
    nn_u = list(nx.non_neighbors(G, u))  # List of non neighbors of u

    if len(nn_u) == 0:
        # if the list is empty, we choose a random node, even if it is neighbor with some node
        # already selected
        print("Impossibile trovare altri nodi non vicini con nessuno")
        G1 = G.deepcopy()
        G1.remove_node(u)    #We can't select u before
        v = random.choice(list(G1.nodes()))
    else:
        # in the list is not empty we create a subgraph containing only the non-neighbors of u and we make
        # a random choice
        G1 = nx.subgraph(G, nn_u)
        v = random.choice(nn_u)

    nn_uv = list(nx.non_neighbors(G1, v))  # List of non neighbors of u and v (if previous list is not empty)

    if len(nn_uv) == 0:
        # if the list is empty, we choose a random node, even if it is neighbor with some node
        # already selected
        print("Impossibile trovare altri nodi non vicini con nessuno")
        G1 = G.deepcopy()
        G1.remove_node(u)
        G1.remove_node(v)
        w = random.choice(list(G1.nodes()))
    else:
        G1 = nx.subgraph(G1, nn_uv)
        w = random.choice(nn_uv)

    nn_uvw = list(nx.non_neighbors(G1, w))  # List of non neighbors of u, v and w (if previous list is not empty)

    if len(nn_uvw) == 0:
        # if the list is empty, we choose a random node, even if it is neighbor with some node
        # already selectioned
        print("Impossibile trovare altri nodi non vicini con nessuno")
        G1 = G.deepcopy()
        G1.remove_node(u)
        G1.remove_node(v)
        G1.remove_node(w)
        z = random.choice(list(G1.nodes()))
    else:
        z = random.choice(nn_uvw)

    print("Initialization phase ended: ", time.time() - start_init_phase)
    return [u, v, w, z]


#This function add to the clusters all the node in samples
#It is called by the naive and parallel implementation of k-means
def cluster_k_means(G, cluster0, cluster1, cluster2, cluster3, samples=None):
    if samples == None:
        #Naive implementation
        samples = G.nodes()
        added = 4
    else:
        #Parallel implementation
        #Because we stop the while when added = n, we have to know how many vertex in samples are
        #already in the four starting cluster (they contain only one vertex)
        added = 0
        if list(cluster0)[0] in samples:
            added+=1
        if list(cluster1)[0] in samples:
            added+=1
        if list(cluster2)[0] in samples:
            added+=1
        if list(cluster3)[0] in samples:
            added+=1

    n = len(samples)

    while added < n:
        # Choose a node that is not yet in a cluster and add it to the closest cluster
        x = random.choice([el for el in samples if el not in cluster0|cluster1|cluster2|cluster3 and (len(
            set(G.neighbors(el)).intersection(cluster0)) != 0 or len(set(G.neighbors(el)).intersection(cluster1)) != 0
            or len(set(G.neighbors(el)).intersection(cluster2)) != 0 or len(set(G.neighbors(el)).intersection(cluster3)) != 0)])

        #Computation of the x's number of neighbors that are in each cluster
        n_c0 = len(set(G.neighbors(x)).intersection(cluster0))
        n_c1 = len(set(G.neighbors(x)).intersection(cluster1))
        n_c2 = len(set(G.neighbors(x)).intersection(cluster2))
        n_c3 = len(set(G.neighbors(x)).intersection(cluster3))

        #we put x in the cluster that contain the highest number of neighbors of x
        if n_c0 != 0 and n_c0 >= n_c1 and n_c0 >= n_c2 and n_c0 >= n_c3:
            cluster0.add(x)
            added+=1
        elif n_c1 != 0 and n_c1 >= n_c2 and n_c1 >= n_c3:
            cluster1.add(x)
            added+=1
        elif n_c2 != 0 and n_c2 >= n_c3:
            cluster2.add(x)
            added+=1
        elif n_c3 != 0:
            cluster3.add(x)
            added+=1

        if added % 250 == 0:
            print(str(added) + " su " + str(n))

    return [cluster0, cluster1, cluster2, cluster3]

#Naive implementation of K-means
def k_means(G):

    #Declaration of initial clusters
    vect = init_phase(G)
    cluster0 = {vect[0]}
    cluster1 = {vect[1]}
    cluster2 = {vect[2]}
    cluster3 = {vect[3]}

    #Computation of clusters
    clusters = cluster_k_means(G, cluster0, cluster1, cluster2, cluster3)

    print(clusters[0], clusters[1], clusters[2], clusters[3])

    #We save each cluster in a different file
    with open("kmeans_result\\cluster0.txt", "w") as f:
        for element in clusters[0]:
            f.write(element + "\n")

    with open("kmeans_result\\cluster1.txt", "w") as f:
        for element in clusters[1]:
            f.write(element + "\n")

    with open("kmeans_result\\cluster2.txt", "w") as f:
        for element in clusters[2]:
            f.write(element + "\n")

    with open("kmeans_result\\cluster3.txt", "w") as f:
        for element in clusters[3]:
            f.write(element + "\n")


#Utility used for split a vector data in chunks of the given size.
#Function used by the parallel implementation
def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}


#Parallel implemention of k-means
def parallel_k_means(G,j):

    # Declaration of initial clusters
    vect = init_phase(G)
    cluster0 = {vect[0]}
    cluster1 = {vect[1]}
    cluster2 = {vect[2]}
    cluster3 = {vect[3]}

    # Declaration of final clusters
    clusters = []
    clusters[0] = set()
    clusters[1] = set()
    clusters[2] = set()
    clusters[3] = set()

    #Computation of clusters using parallelis
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(cluster_k_means)(G, cluster0, cluster1, cluster2, cluster3, X)
                          for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #clusters concatenation of each result
        for res in result:
            clusters[0] = clusters[0] | res[0]
            clusters[1] = clusters[1] | res[1]
            clusters[2] = clusters[2] | res[2]
            clusters[3] = clusters[3] | res[3]

        # We save each cluster in a different file
        with open("kmeans_parallel_result\\cluster0.txt", "w") as f:
            for element in clusters[0]:
                f.write(element + "\n")

        with open("kmeans_parallel_result\\cluster1.txt", "w") as f:
            for element in clusters[1]:
                f.write(element + "\n")

        with open("kmeans_parallel_result\\cluster2.txt", "w") as f:
            for element in clusters[2]:
                f.write(element + "\n")

        with open("kmeans_parallel_result\\cluster3.txt", "w") as f:
            for element in clusters[3]:
                f.write(element + "\n")
