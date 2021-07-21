import networkx as nx
from scipy.sparse import linalg
import numpy as np
from functions import save_clusters

# Naive spectral algorithm
def spectral(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodes).asfptype()

    # computation of n-1 eigenvalues and eigenvectors
    w, v = linalg.eigsh(L, n-1)
    print(w)

    c1 = set()
    c2 = set()
    c3 = set()
    c4 = set()

    # Association of each node to the relative cluster
    for i in range(n):
        if v[i, 0] < 0 and v[i, 1] < 0:
            c1.add(nodes[i])
        elif v[i, 0] < 0:
            c2.add(nodes[i])
        elif v[i, 1] < 0:
            c3.add(nodes[i])
        else:
            c4.add(nodes[i])

    # We save each cluster in a different file
    save_clusters("SPECTRAL/spectral", c1, c2, c3, c4)


# To reduce the graph size, we remove all the nodes with degree equal to 1 because all this node will be part
# of the same cluster of their neighbor. This action reduce the number of nodes from 22470 to 19812
def sampled_spectral(G):
    # SAMPLING PART
    nodes_included = set()
    nodes_excluded = set()
    for node in G.nodes():
        if G.degree(node) > 1:
            nodes_included.add(node)
        else:
            nodes_excluded.add(node)

    G1 = nx.subgraph(G, nodes_included)

    n = G1.number_of_nodes()
    nodes = sorted(G1.nodes())
    L = nx.laplacian_matrix(G1, nodes).asfptype()

    w, v = linalg.eigsh(L, n-1)
    print(w)

    c1 = set()
    c2 = set()
    c3 = set()
    c4 = set()

    for i in range(n):
        if v[i, 0] < 0 and v[i, 1] < 0:
            c1.add(nodes[i])
        elif v[i, 0] < 0:
            c2.add(nodes[i])
        elif v[i, 1] < 0:
            c3.add(nodes[i])
        else:
            c4.add(nodes[i])

    # Add the excluded node to a cluster in which there is his neighbors
    for node in nodes_excluded:
        n_node = next(G.neighbors(node))
        if n_node in c1:
            c1.add(node)
        elif n_node in c2:
            c2.add(node)
        elif n_node in c3:
            c3.add(node)
        elif n_node in c4:
            c4.add(node)
        else:
            print("ERROR")

    # We save each cluster in a different file
    save_clusters("SPECTRAL/sampled_spectral", c1, c2, c3, c4)


# This method is used to get an approximation of the dominant eigenvector of the inverse matrix of L.
# The dominant eigenvector correspond to the eigenvector associated to the largest eigenvalue
# and the dominant eigenvector of the inverse of L correspond to the eigenvector associated to the
# smallest eigenvalue of L
def inverse_power_method(L, n):
    L = L.tocsc().asfptype()
    # The inverse operation of a matrix has a too higher computational and spatial cost
    # For this reason, we calculate and approximation of the inverse using LU decomposition
    A = linalg.spilu(L)
    A = linalg.LinearOperator(shape=(n, n), matvec=A.solve)
    x1 = np.random.rand(n)
    norm = np.linalg.norm(x1)
    x1 = x1/norm

    # using the for loop and calculating the multiplication
    # of power matrix and initial guess and looking for convergence
    for i in range(n):
        x1 = A.dot(x1)
        norm = np.linalg.norm(x1)
        # normalized form
        x1 = x1 / norm

    return x1

# Naive spectral algorithm which uses the inverse power method
def spectral_invpm(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodes).asfptype()

    v = inverse_power_method(L, n)

    cluster1 = set()
    cluster2 = set()

    for i in range(len(v)):
        if v[i] > 0:
            cluster1.add(nodes[i])
        else:
            cluster2.add(nodes[i])

    G1 = nx.subgraph(G, cluster1)
    G2 = nx.subgraph(G, cluster2)
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    nodes1 = sorted(G1.nodes())
    nodes2 = sorted(G2.nodes())
    print(n1)
    print(n2)

    c1 = set()
    c2 = set()
    c3 = set()
    c4 = set()

    # If the resulting cluster are too unbalanced we re-call the inverse_power_method two times only on
    # the biggest cluster, otherwise we re-call the inverse_power_method on both the clusters
    if abs(n1 - n2) < 20000:
        L1 = nx.laplacian_matrix(G1, nodes1).asfptype()
        L2 = nx.laplacian_matrix(G2, nodes2).asfptype()

        v1 = inverse_power_method(L1, n1)
        v2 = inverse_power_method(L2, n2)

        for i in range(len(v1)):
            if v1[i] > 0:
                c1.add(nodes1[i])
            else:
                c2.add(nodes1[i])

        for i in range(len(v2)):
            if v2[i] > 0:
                c3.add(nodes2[i])
            else:
                c4.add(nodes2[i])
    else:
        if n1 > n2:
            c1 = cluster2
            L1 = nx.laplacian_matrix(G1, nodes1).asfptype()
            num = n1
            G_n = G1
            nodes = nodes1
        else:
            c1 = cluster1
            L1 = nx.laplacian_matrix(G2, nodes2).asfptype()
            num = n2
            G_n = G2
            nodes = nodes2


        v = inverse_power_method(L1, num)
        c21 = set()
        c22 = set()
        for i in range(len(v)):
            if v[i] > 0:
                c21.add(nodes[i])
            else:
                c22.add(nodes[i])

        if len(c21) > len(c22):
            c2 = c22
            G_n = nx.subgraph(G_n, c21)
            n_n = G_n.number_of_nodes()
            nodes_n = sorted(G_n.nodes())
            L_n = nx.laplacian_matrix(G_n)
        else:
            c2 = c21
            G_n = nx.subgraph(G_n, c22)
            n_n = G_n.number_of_nodes()
            nodes_n = sorted(G_n.nodes())
            L_n = nx.laplacian_matrix(G_n)

        v = inverse_power_method(L_n, n_n)
        for i in range(len(v)):
            if v[i] > 0:
                c3.add(nodes_n[i])
            else:
                c4.add(nodes_n[i])

    # We save each cluster in a different file
    save_clusters("SPECTRAL/spectral_invpm", c1, c2, c3, c4)


# Same of the spectral_invpm but with sampling
def sampled_spectral_invpm(G):
    nodes_included = set()
    nodes_excluded = set()
    for node in G.nodes():
        if G.degree(node) > 1:
            nodes_included.add(node)
        else:
            nodes_excluded.add(node)

    G1 = nx.subgraph(G, nodes_included)

    n = G1.number_of_nodes()
    nodes = sorted(G1.nodes())
    L = nx.laplacian_matrix(G1, nodes).asfptype()

    v = inverse_power_method(L, n)

    cluster1 = set()
    cluster2 = set()

    for i in range(len(v)):
        if v[i] > 0:
            cluster1.add(nodes[i])
        else:
            cluster2.add(nodes[i])

    G1 = nx.subgraph(G, cluster1)
    G2 = nx.subgraph(G, cluster2)
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    nodes1 = sorted(G1.nodes())
    nodes2 = sorted(G2.nodes())
    print(n1)
    print(n2)

    c1 = set()
    c2 = set()
    c3 = set()
    c4 = set()

    if abs(n1 - n2) < 17500:
        L1 = nx.laplacian_matrix(G1, nodes1).asfptype()
        L2 = nx.laplacian_matrix(G2, nodes2).asfptype()

        v1 = inverse_power_method(L1, n1)
        v2 = inverse_power_method(L2, n2)

        for i in range(len(v1)):
            if v1[i] > 0:
                c1.add(nodes1[i])
            else:
                c2.add(nodes1[i])

        for i in range(len(v2)):
            if v2[i] > 0:
                c3.add(nodes2[i])
            else:
                c4.add(nodes2[i])
    else:
        if n1 > n2:
            c1 = cluster2
            L1 = nx.laplacian_matrix(G1, nodes1).asfptype()
            num = n1
            G_n = G1
            nodes = nodes1
        else:
            c1 = cluster1
            L1 = nx.laplacian_matrix(G2, nodes2).asfptype()
            num = n2
            G_n = G2
            nodes = nodes2

        v = inverse_power_method(L1, num)
        c21 = set()
        c22 = set()
        for i in range(len(v)):
            if v[i] > 0:
                c21.add(nodes[i])
            else:
                c22.add(nodes[i])

        if len(c21) > len(c22):
            c2 = c22
            G_n = nx.subgraph(G_n, c21)
            n_n = G_n.number_of_nodes()
            nodes_n = sorted(G_n.nodes())
            L_n = nx.laplacian_matrix(G_n)
        else:
            c2 = c21
            G_n = nx.subgraph(G_n, c22)
            n_n = G_n.number_of_nodes()
            nodes_n = sorted(G_n.nodes())
            L_n = nx.laplacian_matrix(G_n)

        v = inverse_power_method(L_n, n_n)
        for i in range(len(v)):
            if v[i] > 0:
                c3.add(nodes_n[i])
            else:
                c4.add(nodes_n[i])

    for node in nodes_excluded:
        n_node = next(G.neighbors(node))
        if n_node in c1:
            c1.add(node)
        elif n_node in c2:
            c2.add(node)
        elif n_node in c3:
            c3.add(node)
        elif n_node in c4:
            c4.add(node)
        else:
            print("ERROR")

    # We save each cluster in a different file
    save_clusters("SPECTRAL/sampled_spectral_invpm", c1, c2, c3, c4)