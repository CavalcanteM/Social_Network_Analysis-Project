import networkx as nx

# Implementation of the MinCut algorithm
def MinCut(d0, d1, d2):
    G1 = nx.DiGraph() # In this graph, the deviation of the mean from the value of 3 is considered
    G2 = nx.DiGraph() # In this graph, the deviation of the mean from the value of 2 is considered, separating 1-star samples to 2-star samples
    G3 = nx.DiGraph() # In this graph, the deviation of the mean from the value of 4 is considered, separating 2-star samples to 3-star samples

    # Connection of d0 elements to 's' and 't' nodes
    for x in d0.keys():
        G1.add_edge(x, 't', weight=d0[x])
        G1.add_edge('s', x, weight=1-d0[x])

    # Generation of all the edges with weight infinite for G1
    for x in d0.keys():
        if x[1] != '*' and x[2] != '*':
            node1 = (x[0], x[1], '*')
            node2 = (x[0], '*', x[2])
            G1.add_edge(x, node1, weight=float('inf'))
            G1.add_edge(x, node2, weight=float('inf'))

    # Execution of the first min cut
    # We find two partitions
    # - the first contains all the 1-star samples and part of 2-star samples
    # - the first contains all the 3-star samples and the remaining part of 2-star samples
    _, partitions = nx.algorithms.flow.minimum_cut(G1, 's', 't', capacity='weight')
    # Removing s and t nodes
    partitions[0].remove('s')
    partitions[1].remove('t')

    # Generation of all the edges with weight infinite for G2
    for x in partitions[0]:
        G2.add_edge(x, 't', weight=d1[x])
        G2.add_edge('s', x, weight=1 - d1[x])
        if x[1] != '*' and x[2] != '*':
            node1 = (x[0], x[1], '*')
            node2 = (x[0], '*', x[2])
            if node1 in partitions[0]:
                G2.add_edge(x, node1, weight=float('inf'))
            if node2 in partitions[0]:
                G2.add_edge(x, node2, weight=float('inf'))

    # Generation of all the edges with weight infinite for G3
    for x in partitions[1]:
        G3.add_edge(x, 't', weight=d2[x])
        G3.add_edge('s', x, weight=1 - d2[x])
        if x[1] != '*' and x[2] != '*':
            node1 = (x[0], x[1], '*')
            node2 = (x[0], '*', x[2])
            if node1 in partitions[1]:
                G3.add_edge(x, node1, weight=float('inf'))
            if node2 in partitions[1]:
                G3.add_edge(x, node2, weight=float('inf'))

    # Execution of the second min cut
    # We find two partitions
    # - the first contains all the 1-star samples
    # - the second contains part of 2-star samples
    _, partitions1 = nx.algorithms.flow.minimum_cut(G2, 's', 't', capacity='weight')
    # Execution of the third min cut
    # We find two partitions
    # - the first contains the remaining part of 2-star samples
    # - the second contains all the 3-star samples
    _, partitions2 = nx.algorithms.flow.minimum_cut(G3, 's', 't', capacity='weight')

    # Removing s and t nodes
    partitions1[0].remove('s')
    partitions1[1].remove('t')
    # Removing s and t nodes
    partitions2[0].remove('s')
    partitions2[1].remove('t')

    return partitions1[0], partitions1[1] | partitions2[0], partitions2[1] # Returning the partitions with 1-star samples, 2-star samples and 3-star samples