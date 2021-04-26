import networkx as nx
from scipy.sparse import linalg


#Spectral algorithm version1
def spectral_v1(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodes).asfptype()
    #Laplacian of a graph is a matrix, with diagonal entries being
    # the degree of the corresponding node and off-diagonal entries being -1 if an edge between the corresponding
    # nodes exists and 0 otherwise

    # The following command computes eigenvalues and eigenvectors of the Laplacian matrix.
    # Recall that these are scalar numbers w_1, ..., w_k and vectors v_1, ..., v_k such that Lv_i=w_iv_i.
    # The first output is the array of eigenvalues in increasing order. The second output contains the matrix
    # of eigenvectors: specifically, the eigenvector of the k-th eigenvalue is given by the k-th column of v
    w, v = linalg.eigsh(L, n-1)
    print(w)
    # Partition in clusters based on the corresponding eigenvector value being positive or negative
    # This is known to return (an approximation of) the sparset cut of the graph
    # That is, the cut with each of the clusters having many edges, and with few edge among clusters
    # Note that this is not the minimum cut (that only requires few edge among clusters, but it does not require many edge within clusters)
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
        # We save each cluster in a different file
        with open("SPECTRAL/spectral_result_v1/cluster0.txt", "w") as f:
            for element in c1:
                f.write(element + "\n")

        with open("SPECTRAL/spectral_result_v1/cluster1.txt", "w") as f:
            for element in c2:
                f.write(element + "\n")

        with open("SPECTRAL/spectral_result_v1/cluster2.txt", "w") as f:
            for element in c3:
                f.write(element + "\n")

        with open("SPECTRAL/spectral_result_v1/cluster3.txt", "w") as f:
            for element in c4:
                f.write(element + "\n")


# Spectral algorithm version2
def spectral_v2(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    L = nx.laplacian_matrix(G, nodes).asfptype()
    # Laplacian of a graph is a matrix, with diagonal entries being
    # the degree of the corresponding node and off-diagonal entries being -1 if an edge between the corresponding
    # nodes exists and 0 otherwise

    # The following command computes eigenvalues and eigenvectors of the Laplacian matrix.
    # Recall that these are scalar numbers w_1, ..., w_k and vectors v_1, ..., v_k such that Lv_i=w_iv_i.
    # The first output is the array of eigenvalues in increasing order. The second output contains the matrix
    # of eigenvectors: specifically, the eigenvector of the k-th eigenvalue is given by the k-th column of v
    w, v = linalg.eigsh(L, n - 1)

    # Partition in clusters based on the corresponding eigenvector value being positive or negative
    # This is known to return (an approximation of) the sparset cut of the graph
    # That is, the cut with each of the clusters having many edges, and with few edge among clusters
    # Note that this is not the minimum cut (that only requires few edge among clusters, but it does not require many edge within clusters)
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
        # We save each cluster in a different file
        with open("SPECTRAL/spectral_result_v1/cluster0.txt", "w") as f:
            for element in c1:
                f.write(element + "\n")

        with open("SPECTRAL/spectral_result_v1/cluster1.txt", "w") as f:
            for element in c2:
                f.write(element + "\n")

        with open("SPECTRAL/spectral_result_v1/cluster2.txt", "w") as f:
            for element in c3:
                f.write(element + "\n")

        with open("SPECTRAL/spectral_result_v1/cluster3.txt", "w") as f:
            for element in c4:
                f.write(element + "\n")