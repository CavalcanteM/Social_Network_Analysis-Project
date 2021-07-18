import csv
from priorityq import PriorityQueue
import networkx as nx

def hierarchical(G):
    
    pq = PriorityQueue()
    clusters = set(frozenset([n]) for n in G.nodes())
    for e in G.edges():
        if e[0] != e[1]:                                                            # Cut off self loop
            pq.add( frozenset([frozenset([e[0]]), frozenset([e[1]])]) , 0)

    while len(clusters) >= 5:
        s = list(pq.pop())
        clusters.remove(s[0])
        clusters.remove(s[1])
        
        for w in clusters:
            if frozenset([s[0],w]) in pq.entry_finder:
                pq.remove(frozenset([s[0], w]))
                pq.add(frozenset([s[0]|s[1], w]), max(len(s[0])+len(s[1]),len(w)))
            elif frozenset([w,s[0]]) in pq.entry_finder:
                pq.remove(frozenset([w, s[0]]))
                pq.add(frozenset([s[0]|s[1], w]), max(len(s[0])+len(s[1]),len(w)))
            if frozenset([s[1],w]) in pq.entry_finder:
                pq.remove(frozenset([s[1], w]))
                pq.add(frozenset([s[0]|s[1], w]), max(len(s[0])+len(s[1]),len(w)))
            elif frozenset([w,s[1]]) in pq.entry_finder:
                pq.remove(frozenset([w, s[1]]))
                pq.add(frozenset([s[0]|s[1], w]), max(len(s[0])+len(s[1]),len(w)))

        clusters.add((s[0]|s[1]))
    
    # We save each cluster in a different file
    with open("HIERARCHICAL/optimized/cluster0.txt", "w") as f:
        for element in clusters[0]:
            f.write(element + "\n")

    with open("HIERARCHICAL/optimized/cluster1.txt", "w") as f:
        for element in clusters[1]:
            f.write(element + "\n")

    with open("HIERARCHICAL/optimized/cluster2.txt", "w") as f:
        for element in clusters[2]:
            f.write(element + "\n")

    with open("HIERARCHICAL/optimized/cluster3.txt", "w") as f:
        for element in clusters[3]:
            f.write(element + "\n")

    return clusters
