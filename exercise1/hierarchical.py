import csv
from priorityq import PriorityQueue
import networkx as nx
from functions import save_clusters

def hierarchical(G):
    
    pq = PriorityQueue()
    clusters = set(frozenset([n]) for n in G.nodes())
    for e in G.edges():
        if e[0] != e[1]:                                                            # Cut off self loop
            pq.add( frozenset([frozenset([e[0]]), frozenset([e[1]])]) , 0)

    while len(clusters) >= 5:
        s = list(pq.pop())      #Take the first element from the priority queue, the two elements obtained will be fused into a new cluster
        clusters.remove(s[0])   #Remove from the clusters set the first element obtained
        clusters.remove(s[1])   #Remove from the clusters set the second element obtained
        #Update the distances between clusters in all occurrences where one of the 2 elements obtained appears, and replace the new cluster in its place
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

        clusters.add((s[0]|s[1]))   #Add the new cluster into the set
    
    # We save each cluster in a different file
    save_clusters("HIERARCHICAL/optimized", clusters[0], clusters[1], clusters[2], clusters[3])

    return clusters
