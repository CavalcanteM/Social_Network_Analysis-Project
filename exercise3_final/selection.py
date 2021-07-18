import networkx as nx
from priorityq import PriorityQueue
import math, random
import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it
from tqdm import tqdm
import numpy as np
import time


# Utility used for split a vector data in chunks of the given size.
# Function used by the parallel implementation
def chunks(data, size):
    idata = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in it.islice(idata, size)}


def shapley_degree(G):
    shapley_values = dict()
    list_nodes = list(G.nodes())

    for v in tqdm(range(len(list_nodes)), desc="Calcolo Shapley Degree in corso..."):
        shapley_values[list_nodes[v]] = 1/(1+G.degree(list_nodes[v]))
        for u in G.neighbors(list_nodes[v]):
            shapley_values[list_nodes[v]] += 1/(1+G.degree(u))

    pq = PriorityQueue()
    for node in shapley_values.keys():
        pq.add(node, -shapley_values[node])

    return shapley_values, pq


def shapley_threshold(G, k):
    shapley_values = dict()
    list_nodes = list(G.nodes())

    for v in tqdm(range(len(list_nodes)), desc="Calcolo Shapley Threshold in corso..."):
        shapley_values[list_nodes[v]] = min(1, k / (1 + G.degree(list_nodes[v])))
        for u in G.neighbors(list_nodes[v]):
            shapley_values[list_nodes[v]] += max(0, (G.degree(u) - k + 1) / ((G.degree(u) * (1 + G.degree(u)))))

    pq = PriorityQueue()
    for node in shapley_values.keys():
        pq.add(node, -shapley_values[node])

    return shapley_values, pq


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

    for v in tqdm(range(len(list_nodes)), desc="Calcolo Shapley Closeness in corso..."):
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


def shapley_closeness_parallel(G, j):
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(shapley_closeness)(G, X)
                          for X in chunks(G.nodes(), math.ceil(len(G.nodes()) / j)))

    shapley_values = {v: 0 for v in G.nodes()}
    for res in result:
        for node in res:
            shapley_values[node] += res[node]

    pq = PriorityQueue()
    for node in shapley_values.keys():
        pq.add(node, -shapley_values[node])

    return shapley_values, pq


# The measure associated to each node is exactly its degree divided by number of nodes - 1
def degree(G):
    cen = dict()
    for u in G.nodes():
        cen[u] = G.degree(u) / (G.number_of_nodes() - 1)
    return cen


def sum_bfs(G, u):
    visited = set()
    visited.add(u)
    queue = [u]
    dist = dict()
    dist[u] = 0
    sum = 0

    while len(queue) > 0:
        v = queue.pop(0)
        for w in G[v]:
            if w not in visited:
                visited.add(w)
                queue.append(w)
                dist[w] = dist[v]+1
                sum += dist[w]

    return sum


# It computes the closeness on the subset nodes.
def parallel_closeness(G, nodes):
    closeness = dict()
    for node in nodes:
        closeness[node] = (G.number_of_nodes()-1)/sum_bfs(G, node)
    return closeness


# Parallel implementation of closeness centrality
def parallel_closeness_centrality(G, j):
    with Parallel(n_jobs=j) as parallel:
        result = parallel(delayed(parallel_closeness)(G, X)
                          for X in chunks(G.nodes(), math.ceil(G.number_of_nodes() / j)))

    cl_value = dict()
    pq = PriorityQueue()
    for res in result:
        for x in res.keys():
            cl_value[x] = res[x]
            pq.add(x, -res[x])

    return cl_value, pq



def selector(G, B):
    sp_c, pqs1 = shapley_closeness_parallel(G, 4)
    sp_d, pqs2 = shapley_degree(G)
    sp_t, pqs3 = shapley_threshold(G, 10)
    deg = degree(G)
    cl, pqs4 = parallel_closeness_centrality(G, 4)

    sp_3 = dict()
    sp_deg = dict()
    sp_cl = dict()
    sp_deg_cl = dict()

    max_c = max(sp_c.values())
    min_c = min(sp_c.values())
    for k in sp_c.keys():
        sp_3[k] = (sp_c[k]-min_c)/(max_c-min_c)
        sp_deg[k] = (sp_c[k]-min_c)/(max_c-min_c)
        sp_cl[k] = (sp_c[k]-min_c)/(max_c-min_c)
        sp_deg_cl[k] = (sp_c[k]-min_c)/(max_c-min_c)

    max_d = max(sp_d.values())
    min_d = min(sp_d.values())
    for k in sp_d.keys():
        sp_3[k] += (sp_d[k]-min_d)/(max_d-min_d)
        sp_deg[k] += (sp_d[k]-min_d)/(max_d-min_d)
        sp_cl[k] += (sp_d[k]-min_d)/(max_d-min_d)
        sp_deg_cl[k] += (sp_d[k]-min_d)/(max_d-min_d)

    max_t = max(sp_t.values())
    min_t = min(sp_t.values())
    for k in sp_t.keys():
        sp_3[k] += (sp_t[k]-min_t)/(max_t-min_t)
        sp_deg[k] += (sp_t[k]-min_t)/(max_t-min_t)
        sp_cl[k] += (sp_t[k]-min_t)/(max_t-min_t)
        sp_deg_cl[k] += (sp_t[k]-min_t)/(max_t-min_t)

    max_d = max(deg.values())
    min_d = min(deg.values())
    for k in deg.keys():
        sp_deg[k] += (deg[k]-min_d)/(max_d-min_d)
        sp_deg_cl[k] += (deg[k]-min_d)/(max_d-min_d)

    max_cl = max(cl.values())
    min_cl = min(cl.values())
    for k in cl.keys():
        sp_cl[k] += (cl[k] - min_cl) / (max_cl - min_cl)
        sp_deg_cl[k] += (cl[k] - min_cl) / (max_cl - min_cl)

    pqs5 = PriorityQueue()
    pqs6 = PriorityQueue()
    pqs7 = PriorityQueue()
    pqs8 = PriorityQueue()
    for k in sp_3.keys():
        pqs5.add(k, -sp_3[k])
        pqs6.add(k, -sp_deg[k])
        pqs7.add(k, -sp_cl[k])
        pqs8.add(k, -sp_deg_cl[k])

    # selezioniamo i seed tra le varie Priority Queues
    seeds1 = []     # B shapley closeness
    seeds2 = []     # B shapley degree
    seeds3 = []     # B shapley threshold
    seeds4 = []
    seeds5 = []
    seeds6 = []
    seeds7 = []
    seeds8 = []
    # Riempio seeds1 e seeds2 con B elementi
    for _ in range(B):
        seeds1.append(int(pqs1.pop()))
        seeds2.append(int(pqs2.pop()))
        seeds3.append(int(pqs3.pop()))
        seeds4.append(int(pqs4.pop()))
        seeds5.append(int(pqs5.pop()))
        seeds6.append(int(pqs6.pop()))
        seeds7.append(int(pqs7.pop()))
        seeds8.append(int(pqs8.pop()))

    return [seeds1, seeds2, seeds3, seeds4, seeds5, seeds6, seeds7, seeds8]
