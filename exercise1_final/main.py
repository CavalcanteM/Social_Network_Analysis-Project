import csv
import networkx as nx
import time
from shapley import shapley_degree, shapley_threshold, shapley_closeness
from friedkin import FriedkinJohnsen

def load_graph(filename):

    G = nx.Graph()

    with open(filename, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G


G = load_graph('exercise2_final/nets/net_13')

start = time.time()
print("---------- Shapley Degree ----------")
sp = shapley_degree(G)
print("Calcolo Shapley Degree terminato.")
print("Tempo impiegato: ", time.time()-start)
print()

start = time.time()
print("---------- Shapley Threshold ----------")
k = 1 # parameter for shapley threshold
st = shapley_threshold(G, k)
print("Calcolo Shapley Threshold terminato.")
print("Tempo impiegato: ", time.time()-start)
print()

start = time.time()
print("---------- Shapley Closeness ----------")
sc = shapley_closeness(G)
print("Calcolo Shapley Closeness terminato.")
print("Tempo impiegato: ", time.time()-start)
print()

start = time.time()
print("---------- Friedkin-Johnsen dynamics ----------")
print("Esecuzione Friedkin-Johnsen in corso...")
opinions = FriedkinJohnsen(G)
print("Esecuzione Friedkin-Johnsen terminata.")
print("Tempo impiegato: ", time.time()-start)

# for i in range(17):
#     for j in range(5):
#         G = load_graph('exercise2_final/nets/net_' + str(i+1))
#         start = time.time()
#         print("---------- Friedkin-Johnsen dynamics rete", str(i+1), "- Prova", str(j), "----------")
#         print("Esecuzione Friedkin-Johnsen in corso...")
#         opinions = FriedkinJohnsen(G)
#         print("Esecuzione Friedkin-Johnsen terminata.")
#         print("Tempo impiegato: ", time.time()-start)

# G = load_graph('exercise2_final/nets/net_13')
# values = [0.0]
# for j in values:
#     print("---------- Friedkin-Johnsen dynamics - Stubborness", str(j), "----------")
#     stubborness = {i:j for i in G.nodes()}
#     start = time.time()
#     opinions = FriedkinJohnsen(G, stubborness)
#     print("Tempo impiegato: ", time.time()-start)