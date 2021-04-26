import random
import networkx as nx
import time
from joblib import Parallel, delayed
import itertools as it
import math


# Initial phase in which we add one vertex to each cluster (we assume the presence of only 4 clusters).
# Each vertex is selected from a list of non neighbors of the before selected vertex (if and only if it is possible).
# If the list of non neighbors is empty we select a random and not already selected node from the graph.
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
        G1.remove_node(u)  # We can't select u
        v = random.choice(list(G1.nodes()))
    else:
        # if the list is not empty we create a subgraph containing only the non-neighbors of u and we make
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


# This function add to the clusters all the node in samples
# It is called by the naive and parallel implementation of k-means
def cluster_k_means(G, cluster0, cluster1, cluster2, cluster3, samples=None, parallel=False):
    if samples == None:
        # Naive implementation
        samples = G.nodes()
        added = 4
    elif parallel:
        # Parallel implementation
        # Because we stop the while when added = n, so we have to know how many vertex in samples are
        # already in the four starting cluster (they contain only one vertex
        added = 0
        if list(cluster0)[0] in samples:
            added += 1
        if list(cluster1)[0] in samples:
            added += 1
        if list(cluster2)[0] in samples:
            added += 1
        if list(cluster3)[0] in samples:
            added += 1
    # This case is used in the last part of parallel_k_means, in which we use the naive version because
    # the remaining elements not yet clustered are in small number. In this case, in samples there are only
    # node that are not in a cluster
    else:
        added = 0

    # cluster4 is used only for parallel implementation and in not connected graph.
    # We divided nodes in 4 group and there are a lot of nodes (2658) that have only one edge,
    # so there is an high probability to have a node that is unconnected to the other nodes in his group.
    # This situation causes an exception when we make the random choice because we have added < n
    # while the list on which we make the choice is empty. We add an if len(elements_to_be_clustered)>0 and
    # if it is false we add all the remaining element in the cluster4
    cluster4 = set()

    n = len(samples)

    while added < n:
        # Choose a node that is not yet in a cluster and add it to the closest cluster
        items_to_be_clustered = [el for el in samples if el not in cluster0 | cluster1 | cluster2 | cluster3 and
                            (len(set(G.neighbors(el)).intersection(cluster0)) != 0 or
                            len(set(G.neighbors(el)).intersection(cluster1)) != 0 or
                            len(set(G.neighbors(el)).intersection(cluster2)) != 0 or
                            len(set(G.neighbors(el)).intersection(cluster3)) != 0)]

        # if added < n and len(items_to_be_clustered) == 0 we have some node that can't be added in a cluster
        if len(items_to_be_clustered) > 0:
            x = random.choice(items_to_be_clustered)

            # Computation of the x's number of neighbors that are in each cluster
            n_c0 = len(set(G.neighbors(x)).intersection(cluster0))
            n_c1 = len(set(G.neighbors(x)).intersection(cluster1))
            n_c2 = len(set(G.neighbors(x)).intersection(cluster2))
            n_c3 = len(set(G.neighbors(x)).intersection(cluster3))

            # we put x in the cluster that contain the highest number of neighbors of x
            if n_c0 != 0 and n_c0 >= n_c1 and n_c0 >= n_c2 and n_c0 >= n_c3:
                cluster0.add(x)
                added += 1
            elif n_c1 != 0 and n_c1 >= n_c2 and n_c1 >= n_c3:
                cluster1.add(x)
                added += 1
            elif n_c2 != 0 and n_c2 >= n_c3:
                cluster2.add(x)
                added += 1
            elif n_c3 != 0:
                cluster3.add(x)
                added += 1

            if added % 250 == 0:
                print(str(added) + " su " + str(n))
        else:
            # Element that can't be added to the clusters
            for element in samples:
                if element not in cluster0 | cluster1 | cluster2 | cluster3:
                    cluster4.add(element)

            return [cluster0, cluster1, cluster2, cluster3, cluster4]

    # In this case cluster4 will be empty
    return [cluster0, cluster1, cluster2, cluster3, cluster4]


# Naive implementation of K-means
def k_means(G):
    # Declaration of initial clusters
    vect = init_phase(G)
    cluster0 = {vect[0]}
    cluster1 = {vect[1]}
    cluster2 = {vect[2]}
    cluster3 = {vect[3]}

    # Computation of clusters
    clusters = cluster_k_means(G, cluster0, cluster1, cluster2, cluster3)

    # We save each cluster in a different file
    with open("kmeans_result/cluster0.txt", "w") as f:
        for element in clusters[0]:
            f.write(element + "\n")

    with open("kmeans_result/cluster1.txt", "w") as f:
        for element in clusters[1]:
            f.write(element + "\n")

    with open("kmeans_result/cluster2.txt", "w") as f:
        for element in clusters[2]:
            f.write(element + "\n")

    with open("kmeans_result/cluster3.txt", "w") as f:
        for element in clusters[3]:
            f.write(element + "\n")

    # if the graph is connected len(clusters[4]) will be equal to 0
    if len(clusters[4]) > 0:
        with open("kmeans_result/cluster4.txt", "w") as f:
            for element in clusters[4]:
                f.write(element + "\n")


# Utility used for split a vector data in chunks of the given size.
# Function used by the parallel implementation
def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in it.islice(idata, size)}


# Parallel implemention of k-means
def parallel_k_means(G, j):
    # Declaration of clusters
    vect = init_phase(G)
    cluster0 = {vect[0]}
    cluster1 = {vect[1]}
    cluster2 = {vect[2]}
    cluster3 = {vect[3]}

    done = False
    node_to_cluster = G.nodes()

    # We repeat the operation until all the nodes (or almost all the node if the graph is not connected)
    # are added in a cluster.
    while not done:

        # We ue the parallel version only if we have at least 10*j node to cluster.
        if len(node_to_cluster) > 10*j:
            # Computation of clusters using parallelism
            with Parallel(n_jobs=j) as parallel:
                cluster4 = set()
                result = parallel(delayed(cluster_k_means)(G, cluster0, cluster1, cluster2, cluster3, X, True)
                                  for X in chunks(node_to_cluster, math.ceil(len(node_to_cluster) / j)))
                # clusters concatenation with each result
                for res in result:
                    cluster0 = cluster0 | res[0]
                    cluster1 = cluster1 | res[1]
                    cluster2 = cluster2 | res[2]
                    cluster3 = cluster3 | res[3]
                    cluster4 = cluster4 | res[4]

                if len(cluster4) == 0 or len(cluster4) == len(node_to_cluster):
                    # The cluster algorithm ends
                    done = True
                else:
                    # We put the node without a cluster in 'node_to_cluster'
                    node_to_cluster = G.subgraph(cluster4).nodes()

        else:
            cluster4 = set()
            # We call the naive k-means algorithm to classify the remaining element because with
            # a small number of samples is not convenient use the parallel version
            result = cluster_k_means(G, cluster0, cluster1, cluster2, cluster3, cluster4)
            done = True
            cluster0 = cluster0 | result[0]
            cluster1 = cluster1 | result[1]
            cluster2 = cluster2 | result[2]
            cluster3 = cluster3 | result[3]
            cluster4 = cluster4 | result[4]

    # We save each cluster in a different file
    with open("kmeans_parallel_result/cluster0.txt", "w") as f:
        for element in cluster0:
            f.write(element + "\n")
    with open("kmeans_parallel_result/cluster1.txt", "w") as f:
        for element in cluster1:
            f.write(element + "\n")
    with open("kmeans_parallel_result/cluster2.txt", "w") as f:
        for element in cluster2:
            f.write(element + "\n")
    with open("kmeans_parallel_result/cluster3.txt", "w") as f:
        for element in cluster3:
            f.write(element + "\n")

    # cluster4 contains only the element that we can't add in cluster (not connected graph)
    if len(cluster4) > 0:
        with open("kmeans_parallel_result/cluster4.txt", "w") as f:
            for element in cluster4:
                f.write(element + "\n")



# Optimized version of k-means. (This function is also called by the parallel and optimized version)
# The bottle neck of this problem is the intersection operation between set and the for cycle used to
# build the list items_to_be_clustered. To remove this bottle neck, we use two new set:
#
# samples: it contains all the elements not added in a cluster yet
# neighbors_cluster: it contains all the node in the clusters and their neighbors
#
# It can be proved that the intersection between this two set is equivalent to the for cycle at line 102,
# reducing the complexity of that operation from O(n*min(degree(el),len(cluster)) to O(min(len(cluster),len(samples)).
# Because we remove from samples each element added to a cluster, we have:
# at the start of the algorithm len(samples) = n-4, but len(cluster) = 1 -> O(min(len(cluster),len(samples)) = O(1)
# at the end, in the worst case, len(cluster) = n-4 , but len(samples) = 1 -> O(min(len(cluster),len(samples)) = 0(1)
# in the middle of the algorithm, in the worst case, len(cluster) = n/2 and len(samples)=n/2 ---->
# ----> O(min(len(cluster),len(samples))  = 0(n/2)
# So we have, O(min(len(cluster),len(samples)) << O(n*min(degree(el),len(cluster))
# N.B. These operations are repeated n times.
def opt_cluster_k_means(G, cluster0, cluster1, cluster2, cluster3, elements, parallel = False):
    if parallel:
        elements = set(elements)
    # Set of clustered nodes and their neighbors
    neighbors_clusters = cluster0 | cluster1 | cluster2 | cluster3
    for x in cluster0:
        neighbors_clusters = neighbors_clusters | set(G.neighbors(x))
    for x in cluster1:
        neighbors_clusters = neighbors_clusters | set(G.neighbors(x))
    for x in cluster2:
        neighbors_clusters = neighbors_clusters | set(G.neighbors(x))
    for x in cluster3:
        neighbors_clusters = neighbors_clusters | set(G.neighbors(x))

    while True:
        # Choose a  random node that is not yet in a cluster, using only one intersection operation
        # instead of n intersection operation
        items_to_be_clustered = list(elements.intersection(neighbors_clusters))
        if len(items_to_be_clustered) == 0:
            break
        x = random.choice(items_to_be_clustered)
        elements.remove(x)

        # Computation of the x's number of neighbors that are in each cluster
        n_c0 = len(set(G.neighbors(x)).intersection(cluster0))
        n_c1 = len(set(G.neighbors(x)).intersection(cluster1))
        n_c2 = len(set(G.neighbors(x)).intersection(cluster2))
        n_c3 = len(set(G.neighbors(x)).intersection(cluster3))

        # we put x in the cluster that contain the highest number of neighbors of x
        if n_c0 != 0 and n_c0 >= n_c1 and n_c0 >= n_c2 and n_c0 >= n_c3:
            cluster0.add(x)

        elif n_c1 != 0 and n_c1 >= n_c2 and n_c1 >= n_c3:
            cluster1.add(x)

        elif n_c2 != 0 and n_c2 >= n_c3:
            cluster2.add(x)
        else:
            cluster3.add(x)

        # We add in neighbors_clusters x and all its neighbors
        neighbors_clusters.add(x)
        neighbors_clusters = neighbors_clusters | set(G.neighbors(x))

    return cluster0, cluster1, cluster2, cluster3, elements



# Optimized version of k-means
def optimized_k_means(G):
    # Declaration of initial clusters
    vect = init_phase(G)
    cluster0 = {vect[0]}
    cluster1 = {vect[1]}
    cluster2 = {vect[2]}
    cluster3 = {vect[3]}
    cluster4 = set()

    # All the samples not added in the solution yet
    samples = set(G.nodes())
    for x in vect:
        samples.remove(x)

    (cluster0, cluster1, cluster2, cluster3, cluster4) = opt_cluster_k_means(G, cluster0, cluster1, cluster2, cluster3, samples)

    # We save each cluster in a different file
    with open("optimized_kmeans_result/cluster0.txt", "w") as f:
        for element in cluster0:
            f.write(element + "\n")

    with open("optimized_kmeans_result/cluster1.txt", "w") as f:
        for element in cluster1:
            f.write(element + "\n")

    with open("optimized_kmeans_result/cluster2.txt", "w") as f:
        for element in cluster2:
            f.write(element + "\n")

    with open("optimized_kmeans_result/cluster3.txt", "w") as f:
        for element in cluster3:
            f.write(element + "\n")

    # cluster4 contains only the element that we can't add in cluster (not connected graph)
    if len(cluster4) > 0:
        with open("optimized_kmeans_result/cluster4.txt", "w") as f:
            for element in cluster4:
                f.write(element + "\n")



# The optimized version implemented in a parallel way
def parallel_opt_k_means(G,j):
    # Declaration of initial clusters
    vect = init_phase(G)
    cluster0 = {vect[0]}
    cluster1 = {vect[1]}
    cluster2 = {vect[2]}
    cluster3 = {vect[3]}

    # All the samples not added in the solution yet
    samples = set(G.nodes())
    for x in vect:
        samples.remove(x)
    i = 1

    # We repeat the operation until all the nodes or almost all the node are added in a cluster.
    while len(samples) > 40*j:
        with Parallel(n_jobs=j) as parallel:
            samples_to_cluster = G.subgraph(samples).nodes()
            samples = set()

            result = parallel(delayed(opt_cluster_k_means)(G, cluster0, cluster1, cluster2, cluster3, X, True)
                              for X in chunks(samples_to_cluster, math.ceil(len(samples_to_cluster) / j)))

            for res in result:
                cluster0 = cluster0 | res[0]
                cluster1 = cluster1 | res[1]
                cluster2 = cluster2 | res[2]
                cluster3 = cluster3 | res[3]
                samples = samples | res[4]
                for r in res:
                    print(len(r))
                print("\n")

    if len(samples) > 0:
        (cluster0, cluster1, cluster2, cluster3, samples) = opt_cluster_k_means(G, cluster0, cluster1, cluster2, cluster3, samples)


    # We save each cluster in a different file
    with open("parallel_optimized_k_means_result/cluster0.txt", "w") as f:
        for element in cluster0:
            f.write(element + "\n")

    with open("parallel_optimized_k_means_result/cluster1.txt", "w") as f:
        for element in cluster1:
            f.write(element + "\n")

    with open("parallel_optimized_k_means_result/cluster2.txt", "w") as f:
        for element in cluster2:
            f.write(element + "\n")

    with open("parallel_optimized_k_means_result/cluster3.txt", "w") as f:
        for element in cluster3:
            f.write(element + "\n")

    # cluster4 contains only the element that we can't add in cluster (not connected graph)
    if len(samples) > 0:
        with open("parallel_optimized_k_means_result/cluster4.txt", "w") as f:
            for element in samples:
                f.write(element + "\n")