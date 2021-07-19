import csv
import networkx as nx
import random
from manipulator import manipulation
import time


def random_vector(n):
    v = []
    for _ in range(n):
        v.append(random.uniform(0, 1))
    return v


def load_graph():

    G = nx.Graph()

    with open('../exercise2_final/nets/net_13', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ')

        for row in rows:
            G.add_edge(row[0], row[1])

    return G


p = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]   # Vettore dei candidati
c = random.randint(0, len(p)-1)     # Candidato da favorire
B = 200     # Numero di seed da selezionare
b = random_vector(10000)     # Vettore delle single-peaked preference
G = load_graph()

start = time.time()
manipulation(G, p, c, B, b)
end = time.time() - start
print(end)
