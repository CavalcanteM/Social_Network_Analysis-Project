import networkx as nx

# Implementation of the Shapley_degree centrality measure
# It is the Shapley value for the characteristic function
# value(C) = |C| + |N(C)|
# where N(C) is the set of nodes outside C with at least one neighbor in C
def shapley_degree(G):
    pass

# Implementation of the Shapley_threshold centrality measure
# It is the Shapley value for the characteristic function
# value(C) = |C| + |N(C,k)|
# where N(C,k) is the set of nodes outside C with at least k neighbors in C
def shapley_threshold(G, k):
    pass

# Implementation of the Shapley_threshold centrality measure
# It is the Shapley value for the characteristic function
# value(C) = Sum(1/dist(u,C))
# where dist(u,C) is the minimum distance between u and a node of C
def shapley_closeness(G, u):
    pass