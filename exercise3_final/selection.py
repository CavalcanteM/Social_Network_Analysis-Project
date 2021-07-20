import math
import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
from exercise1_final.shapley import shapley_degree, shapley_threshold


# Utility used for split a vector data in chunks of the given size.
# Function used by the parallel implementation
def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in it.islice(idata, size)}


def bfs(G, u):
    visited = set()  # nodi visitati
    visited.add(u)
    queue = [u]
    dist = dict()  # dizionario nodo:distanza
    dist[u] = 0

    while len(queue) > 0:
        v = queue.pop(0)
        for w in G[v]:
            if w not in visited:
                visited.add(w)
                queue.append(w)
                dist[w] = dist[v] + 1

    sort_dist = sorted(dist.items(), key=lambda x: x[1])

    nodes = []
    distances = []
    for i in sort_dist:
        nodes.append(i[0])
        distances.append(i[1])
    return nodes, distances


def shapley_closeness(G, nodes):
    shapley_values = {v: 0 for v in G.nodes()}
    list_nodes = list(nodes)

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


# Parallel implementation of Shapley Closeness
def shapley_closeness_parallel(G, j):
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(shapley_closeness)(G, X)
                          for X in chunks(G.nodes(), math.ceil(len(G.nodes()) / j)))

    shapley_values = {v: 0 for v in G.nodes()}
    for res in result:
        for node in res:
            shapley_values[node] += res[node]

    return shapley_values


# Function that perform the selection of B nodes in according to the shapley closeness, shapley degree and shapley
# threshold.
def selector(G, B):
    # Compute the various shapley value
    sh_cl = shapley_closeness_parallel(G, 4)
    sh_d = shapley_degree(G)
    sh_t = shapley_threshold(G, 5)

    # final dict that will contain the sum of the normalized shapley values
    final_dict = dict()

    max_c = max(sh_cl.values())
    min_c = min(sh_cl.values())
    max_d = max(sh_d.values())
    min_d = min(sh_d.values())
    max_t = max(sh_t.values())
    min_t = min(sh_t.values())
    for k in sh_cl.keys():
        final_dict[k] = (sh_cl[k]-min_c)/(max_c-min_c)
        final_dict[k] += (sh_d[k]-min_d)/(max_d-min_d)
        final_dict[k] += (sh_t[k]-min_t)/(max_t-min_t)

    pq = PriorityQueue()
    for k in final_dict.keys():
        pq.add(k, -final_dict[k])

    # selection phase
    seeds = []
    while len(seeds) < B:
        seeds.append(int(pq.pop()))

    return seeds
