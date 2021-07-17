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

    with open('../exercise2_final/nets/net_' + str(x), newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ')
        #next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G


p = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]   # Vettore dei candidati
#c = random.randint(0, len(p)-1)     # Candidato da favorire
c = 2
B = 200     # Numero di seed da selezionare
b = random_vector(10000)     # Vettore delle single-peaked preference
print(p[c])

for i in range(1, 18):
    if i != 14:
        G = load_graph(i)
        start = time.time()
        res = manipulation(G, p, c, B, b)
        end = time.time() - start
        print(end)
        with open("result5.txt", "a") as f:
            f.write("Rete" + str(i) + " con Shapley Closeness:\n")
            f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[1]) + "\n")
            f.write("------------------------------------------------------\n")
            f.write("Rete" + str(i) + " con Shapley Degree:\n")
            f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[2]) + "\n")
            f.write("------------------------------------------------------\n")
            f.write("Rete" + str(i) + " con Shapley Threshold(k=10):\n")
            f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[3]) + "\n")
            f.write("------------------------------------------------------\n")
            f.write("Rete" + str(i) + " con Closeness:\n")
            f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[4]) + "\n")
            f.write("------------------------------------------------------\n")
            f.write("Rete" + str(i) + " con CLOSENESS+DEGREE+THRESHOLD:\n")
            f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[5]) + "\n")
            f.write("------------------------------------------------------\n")
            f.write("Rete" + str(i) + " con CLOSENESS+DEGREE+THRESHOLD+NORMAL_DEGREE:\n")
            f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[6]) + "\n")
            f.write("------------------------------------------------------\n")
            f.write("Rete" + str(i) + " con CLOSENESS+DEGREE+THRESHOLD+NORMAL_CLOSENESS:\n")
            f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[7]) + "\n")
            f.write("------------------------------------------------------\n")
            f.write("Rete" + str(i) + " con CLOSENESS+DEGREE+THRESHOLD+NORMAL_DEGREE_CLOSENESS:\n")
            f.write("Risultati truthful: " + str(res[0]) + "   Risultati manipolati: " + str(res[8]) + "\n")
            f.write("------------------------------------------------------\n\n")
