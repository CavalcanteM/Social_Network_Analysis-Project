import csv
import networkx as nx
import matplotlib.pyplot as plt
# import sys
# sys.path.append('c:\\Users\\danya\\Desktop\\SNA-Project\\Exercise 2')
# import degree
from lesson4 import randomG

def load_graph():

    G = nx.Graph()

    with open('exercise2_final/nets/net_13', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G

def degree(G, nodes):
    cen = dict()
    for u in nodes:
        cen[u] = G.degree(u)
    return cen

G = load_graph()
# print("Number of nodes: ", G.number_of_nodes())
n = 10000
p = 0.007
G1 = randomG(n, p)
# Compute degree centrality
dd = degree(G, G.nodes())
dd1 = degree(G1, G1.nodes())
# print("Min value: ", min(dc_dict.values()))
# print("Max value: ", max(dc_dict.values()))

# Plot degree centrality distribution
plt.hist(dd.values())
plt.title("Degree distribution")
plt.xlabel("Value")
plt.ylabel("Number of nodes")
plt.show()

# Plot degree centrality distribution
plt.hist(dd1.values())
plt.title("Generated Degree distribution")
plt.xlabel("Value")
plt.ylabel("Number of nodes")
plt.show()