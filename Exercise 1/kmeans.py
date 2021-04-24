import random
import networkx as nx
import time

def k_means(G):
    start_init_phase = time.time()

    n = G.number_of_nodes()
    # Choose four clusters represented by vertices that are not neighbors

    u = random.choice(list(G.nodes()))
    nn_u = list(nx.non_neighbors(G, u))     # List of non neighbors of u

    if len(nn_u) == 0:
        # if the list is empty, we choose a random node, even if it is neighbor with some node
        # already selected
        print("Impossibile trovare altri nodi non vicini con nessuno")
        G1 = G.deepcopy()
        G1.remove_node(u)
        v = random.choice(list(G1.nodes()))
    else:
        # in the list is not empty we create a subgraph containing only the non-neighbors of u and we make
        # a random choice
        G1 = nx.subgraph(G, nn_u)
        v = random.choice(nn_u)

    nn_uv = list(nx.non_neighbors(G1,v))     # List of non neighbors of u and v (if previous list is not empty)

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

    nn_uvw = list(nx.non_neighbors(G1,w))     # List of non neighbors of u, v and w (if previous list is not empty)

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
        G1 = nx.subgraph(G1, nn_uvw)
        z = random.choice(nn_uvw)


    cluster0 = {u}
    cluster1 = {v}
    cluster2 = {w}
    cluster3 = {z}
    added = 4

    print("Initialization phase ended: ", time.time()-start_init_phase)

    while added < n:
        # Choose a node that is not yet in a cluster and add it to the closest cluster
        x = random.choice([el for el in G.nodes() if el not in cluster0|cluster1|cluster2|cluster3 and (len(
            set(G.neighbors(el)).intersection(cluster0)) != 0 or len(set(G.neighbors(el)).intersection(cluster1)) != 0
            or len(set(G.neighbors(el)).intersection(cluster2)) != 0 or len(set(G.neighbors(el)).intersection(cluster3)) != 0)])

        n_c0 = len(set(G.neighbors(x)).intersection(cluster0))
        n_c1 = len(set(G.neighbors(x)).intersection(cluster1))
        n_c2 = len(set(G.neighbors(x)).intersection(cluster2))
        n_c3 = len(set(G.neighbors(x)).intersection(cluster3))

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

    print(cluster0, cluster1, cluster2, cluster3)

    with open("kmeans_result\\cluster0.txt", "w") as f:
        for element in cluster0:
            f.write(element + "\n")

    with open("kmeans_result\\cluster1.txt", "w") as f:
        for element in cluster1:
            f.write(element + "\n")

    with open("kmeans_result\\cluster2.txt", "w") as f:
        for element in cluster2:
            f.write(element + "\n")

    with open("kmeans_result\\cluster3.txt", "w") as f:
        for element in cluster3:
            f.write(element + "\n")