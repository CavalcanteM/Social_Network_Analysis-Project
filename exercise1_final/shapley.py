import networkx as nx
# from tqdm import tqdm

# Polynomial-time algorithm to compute the Shapley value based on the characteristic function 'shapley degree'
# value(C) = |C| + |N(C)| where N(C) is the set of nodes outside C with at least one neighbor in C.
# To evaluate the Shapley value of node vi, we should consider all possible permutations of the nodes in which vi
# would make a positive marginal contribution to the coalition of nodes occurring before itself.
# Instead of compute all possible permutations, is proved that vi gives a marginally contribute to his neighbor vj
# to fringe C when vj and his neighbors are not in C.
# So, the probability that in a random permutation none of the vertices from neighbors(vj)∪{vj} occurs before vi,
# where vi and vj are neighbours, is represented as a Bernoulli random variable with expected value 1/(1+degG(vj)).
# The shapley value, for this characteristic function, of a node vi can be written as the sum of 1/(1+deg(v))
# for each v in neighbors(vi)∪{vi}
# Complexity O(n+m)
def shapley_degree(G):
    shapley_values = dict()
    list_nodes = list(G.nodes())

    # for v in tqdm(range(len(list_nodes)), desc="Calcolo Shapley Degree in corso..."):
    for v in range(len(list_nodes)):
        shapley_values[list_nodes[v]] = 1/(1+G.degree(list_nodes[v]))
        for u in G.neighbors(list_nodes[v]):
            shapley_values[list_nodes[v]] += 1/(1+G.degree(u))

    return shapley_values


# Polynomial-time algorithm to compute the Shapley value based on the characteristic function 'shapley threshold'
# value(C) = |C| + |N(C,k)| where N(C,k) is the set of nodes outside C with at least k neighbors in C.
# The reasoning is similar to that of shapley degree, but if deg(vj) < k, we have E[Bvi,vj] = 1 for vi = vj and 0
# otherwise. For degree(vj) ≥ k, we split the argument into two cases. If vj != vi, the condition for marginal
# contribution is that exactly (k − 1) neighbors of vj already belong to Ci and vj ∈/ Ci.
# On the other hand, if vj = vi, the marginal contribution occurs if and only if Ci originally consisted of at
# most (k − 1) neighbors of vj . So for degree(vj) ≥ k and vj != vi, is proved that the probability that in a random
# permutation exactly k−1 neighbours of vj occur before vi, and vj occurs after vi, is:
# (1+degree(vj)−k)/degG(vj )(1+degG(vj )), where vj and vi are neighbors and degG(vj) ≥ k.
# The shapley value, for this characteristic function, of a node vi can be written as the sum of the expected value
# of the Bernoulli function for each v in neighbors(vi)∪{vi}.
# Complexity O(n+m)
def shapley_threshold(G, k):
    shapley_values = dict()
    list_nodes = list(G.nodes())
    
    # for v in tqdm(range(len(list_nodes)), desc="Calcolo Shapley Threshold in corso..."):
    for v in range(len(list_nodes)):
        shapley_values[list_nodes[v]] = min(1, k/(1+G.degree(list_nodes[v])))
        for u in G.neighbors(list_nodes[v]):
            shapley_values[list_nodes[v]] += max(0, (G.degree(u)-k+1)/((G.degree(u)*(1+G.degree(u)))))

    return shapley_values

def bfs(G, u):
    visited = set() # nodi visitati
    visited.add(u)
    queue = [u]
    dist = dict()   # dizionario nodo:distanza
    dist[u] = 0

    while len(queue) > 0:
        v = queue.pop(0)
        for w in G[v]:
            if w not in visited:
                visited.add(w)
                queue.append(w)
                dist[w] = dist[v]+1

    sort_dist = sorted(dist.items(), key=lambda x: x[1])
    
    nodes = []
    distances = []
    for i in sort_dist:
        nodes.append(i[0])
        distances.append(i[1])
    return nodes, distances

# Polynomial-time algorithm to compute the Shapley value based on the characteristic function 'shapley closeness'
# value(C) = Sum(1/dist(u,C)) where dist(u,C) is the minimum distance between u and a node of C.
# Instead of compute all possible permutations, is proved that the probability that in a random permutation none of the
# nodes from {vj , w1, . . . , wk} occur before vi and the node wk+1 occurs before vi is 1/(k+1)(k+2).
# Complexity O(n*m + (n^2)*log(n))
def shapley_closeness(G):
    shapley_values = {v:0 for v in G.nodes()}
    list_nodes = list(G.nodes())
    
    # for v in tqdm(range(len(list_nodes)), desc="Calcolo Shapley Closeness in corso..."):
    for v in range(len(list_nodes)):
        nodes, distances = bfs(G, list_nodes[v])
        sum = 0
        index = G.number_of_nodes() - 1
        prevDistance = -1
        prevSV = -1

        while index > 0:
            if distances[index] == prevDistance:
                currSV = prevSV
            else:
                currSV = 1 / (distances[index] * (1 + index)) - sum

            shapley_values[nodes[index]] += currSV
            sum += 1 / (distances[index] * (index * (1 + index)))
            prevDistance = distances[index]
            prevSV = currSV
            index -= 1

        shapley_values[list_nodes[v]] += 1 - sum

    return shapley_values
