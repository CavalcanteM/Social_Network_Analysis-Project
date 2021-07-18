import os, csv, math
import networkx as nx
from priorityq import PriorityQueue
from joblib import Parallel, delayed
import itertools as it

def hierarchical(G, X):
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in X:
        for v in X:
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():
                    pq.add(frozenset([frozenset([u]), frozenset([v])]), 0)  
                else:
                    pq.add(frozenset([frozenset([u]), frozenset([v])]), 1)
    print("Read done")
    # Start with a cluster for each node
    clusters = set(frozenset([u]) for u in X)
    done = False
    while not done:
        # Merge closest clusters
        s = list(pq.pop())
        clusters.remove(s[0])
        clusters.remove(s[1])

        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset([s[0] | s[1], w]), 0)
            else:
                pq.add(frozenset([s[0] | s[1], w]), 1)

        clusters.add(s[0] | s[1])

        if len(clusters)<5:
            print("Clustering done")
            done = True

    return clusters


def chunks(data,size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}


def parallel_hier(G,j):
    #Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        
        results = parallel(delayed(hierarchical)(G,X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))
        #Aggregates the results
        clusters = list()
        for result in results:
            for cluster in result:
                clusters.append(cluster)
        results=[]
        print(clusters)
        flag = False
        while len(clusters)>4:
            flag = False
            for a in range(len(clusters)):
                if flag == True:
                    break
                for b in range(len(clusters)):
                    if flag == True:
                        break
                    if a != b:
                        if merge_clusters(G,clusters[a],clusters[b]):
                            s1 = clusters.pop(a)
                            if a<b:
                                s2 = clusters.pop(b-1)
                            else:
                                s2 = clusters.pop(b)
                            clusters.append(s1|s2)
                            flag = True
    return clusters

def merge_clusters(G,c1,c2):
    links=0
    for u in c1:
        for v in c2:
            if (u, v) in G.edges() or (v, u) in G.edges():
                links += 1
    if links>(len(c1)/4) or links>(len(c2)/4):
        return True
    return False

os.chdir("facebook_large")

G = nx.Graph()

with open('musae_facebook_edges.csv', newline='') as csvfile:

    rows = csv.reader(csvfile, delimiter=',')
    next(rows)
    
    for row in rows:
      G.add_edge(row[0], row[1])

clusters=parallel_hier(G,16)

with open('hier_parallel.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    i=0
    for cluster in clusters:
        i+=1
        for node in cluster:
            spamwriter.writerow([node,i])
