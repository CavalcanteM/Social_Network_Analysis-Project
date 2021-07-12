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


def load_graph(x):

    G = nx.Graph()

    with open('../Exercise 2 Final/nets/net_' + str(x), newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G


with open("result.txt", "w") as f:
    p = random_vector(10)   # Vettore dei candidati
    c = random.randint(0, len(p)-1)     # Candidato da favorire
    B = 250     # Numero di seed da selezionare
    b = random_vector(10000)     # Vettore delle single-peaked preference
    for i in range(1, 18):
        G = load_graph(i)
        start = time.time()
        res = manipulation(G, p, c, B, b)
        end = time.time() - start
        print(end)
        f.write("Rete" + str(i) + " con Girman-Newman(4 cluster)+HITS su sottografi:\n")
        f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[1]) + "\n\n")
        f.write("------------------------------------------------------")
        f.write("Rete" + str(i) + " con Girman-Newman(4 cluster)+Pagerank su sottografi:\n")
        f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[2]) + "\n\n")
        f.write("------------------------------------------------------")
        f.write("Rete" + str(i) + " con Girman-Newman(4 cluster)+Degree su sottografi:\n")
        f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[3]) + "\n\n")
