import csv
import networkx as nx

def load_graph():

    G = nx.Graph()

    with open('../nets/net_13', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ')
        next(rows)

        for row in rows:
            G.add_edge(row[0], row[1])

    return G


G = load_graph()