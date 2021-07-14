import csv
import networkx as nx
import time
from shapley import shapley_degree, shapley_threshold, shapley_closeness
from friedkin import FriedkinJohnsen

def load_graph():

    G = nx.Graph()

    with open('exercise1_final/net_13', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G


G = load_graph()

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