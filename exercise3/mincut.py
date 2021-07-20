import networkx as nx

def MinCut(d1, d2):
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    G3 = nx.DiGraph()
    for x in d1.keys():
        G1.add_edge(x, 't', weight=d1[x])
        G1.add_edge('s', x, weight=1-d1[x])

    for x in d1.keys():
        if x[1] != '*' and x[2] != '*':
            node1 = (x[0], x[1], '*')
            node2 = (x[0], '*', x[2])
            G1.add_edge(x, node1, weight=float('inf'))
            G1.add_edge(x, node2, weight=float('inf'))

    value, partitions = nx.algorithms.flow.minimum_cut(G1, 's', 't', capacity='weight')
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

    value1, partitions1 = nx.algorithms.flow.minimum_cut(G2, 's', 't', capacity='weight')
    value2, partitions2 = nx.algorithms.flow.minimum_cut(G3, 's', 't', capacity='weight')

    return partitions1[0], partitions1[1] | partitions2[0], partitions2[1]