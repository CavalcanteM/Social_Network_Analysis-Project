import networkx as nx

def MinCut(d0, d1, d2):
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    G3 = nx.DiGraph()
    for x in d0.keys():
        G1.add_edge(x, 't', weight=d0[x])
        G1.add_edge('s', x, weight=1-d0[x])

    for x in d0.keys():
        if x[1] != '*' and x[2] != '*':
            node1 = (x[0], x[1], '*')
            node2 = (x[0], '*', x[2])
            G1.add_edge(x, node1, weight=float('inf'))
            G1.add_edge(x, node2, weight=float('inf'))

    _, partitions = nx.algorithms.flow.minimum_cut(G1, 's', 't', capacity='weight')
    partitions[0].remove('s')
    partitions[1].remove('t')

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

    _, partitions1 = nx.algorithms.flow.minimum_cut(G2, 's', 't', capacity='weight')
    _, partitions2 = nx.algorithms.flow.minimum_cut(G3, 's', 't', capacity='weight')

    partitions1[0].remove('s')
    partitions1[1].remove('t')

    partitions2[0].remove('s')
    partitions2[1].remove('t')

    return partitions1[0], partitions1[1] | partitions2[0], partitions2[1]