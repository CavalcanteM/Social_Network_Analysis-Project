#!/usr/bin/python

import networkx as nx
import random
from scipy.special import zeta
import math
import numpy

#Random Graph (Newman, chap. 12)
#n = number of nodes
#p = probability of inserting an edge
def randomG(n, p):
    G = nx.Graph()
    for i in range(n):
        for j in range (i+1, n):
            r=random.random()
            if r <= p:
                G.add_edge(i,j)
    return G
#Low clustering coefficient
#If p > 1/n-1, there is high probability that there will be a giant component
#The expected diameter is ln n

#Configuration Model (Newman, chap. 13)
#Random Graph with a given degree sequence deg.
#We assume that deg is a valid degree sequence (i.e., the sum of degrees is even) and that no node has degree 0.
def configurationG(deg):
    G = nx.Graph()
    # The following list will contain all nodes for which there is at least another neighbor to add
    nodes=list(range(len(deg)))
    # We consider '> 1' and not '>0' because when len(nodes) = 1 only loops are possible
    # Hence, the degree sequence of the resulting graph may not be exactly the same as the one in input for the single remaining node.
    # However, note that this single "outlier" does not alter the degree sequence distribution.
    while len(nodes)>1:
        edge=random.sample(nodes,2)
        if not G.has_edge(edge[0],edge[1]):
            G.add_edge(edge[0],edge[1])
            deg[edge[0]]-=1
            if deg[edge[0]] == 0:
                nodes.remove(edge[0])
            deg[edge[1]]-=1
            if deg[edge[1]] == 0:
                nodes.remove(edge[1])
    return G
#Most of the properties depends on the degree distribution: fraction p_k of vertices with degree k
#Usually, clustering coefficient is low. However, it becomes larger when the degree distribution is a power law
#Usually, configuration graph have a giant component.
#For power law degree distribution, the giant component usually exists for low values of the exponent

#It returns a degree sequence that follows a power law
#n = number of nodes
#power = the exponent of the power law distribution
def power_law_degree(n,power):
    deg = 1 #The smallest degree in the degree sequence
    deg_list=[] #The degree sequence
    # In order to follow a power law, the fraction of times that degree d appears in the degree sequence
    # must be 1/d**power. Since the sum of these fractions must be 1, we normalize each fraction.
    # I.e., d appears in the sequence a fraction of times that is (1/d**power)/Z,
    # where Z = \sum_d 1/d**power. The value of Z is computed by the function zeta in the library scipy.special
    z=zeta(power)
    somma=0
    while len(deg_list) < n:
        p = 1/((deg**power)*z) #The fraction of occurrences of deg in the sequence
        num=math.ceil(p*n) #The number of occurrences of deg in the sequence.
        #We may use the floor or the closest integer. However, be careful:
        #If we use math.floor, then when the fraction is very small the number returned is 0,
        #thus no degree is inserted in the sequence, and the algorithm does not terminate

        # Add deg to the sequence num times, or until the sequence is not full
        for i in range(num):
            if len(deg_list) == n:
                break
            deg_list.append(deg)
            somma+=deg
        deg+=1

    # To check that the generated sequence is valid, one can compute the sum of the inserted degree
    # (this is done within the above while loop)
    # If the value of the sum is odd, to fix the sequence, it is sufficient to increase the value of deg_list[0].
    if somma %2 != 0:
        deg_list[0]+=1

    return deg_list

#Preferential Attachment (EK 18)
#n=nodes
#p=probability
#Nodes comes one at the time.
#With probability p, they will choose their neighbor with a probability proportional to their degree
#(nodes with higher degree are chosen with larger probability),
#with the remaining probability, a neighbor is chosen uniformly at random.
def preferentialG(n,p):
    G = nx.Graph()
    nodes=[] #Keep as many copies of a node as the degree of that node
    for u in range(n):
        r=random.random()
        if r <= p and len(nodes) > 0: #For the first node preferential attachment cannot be executed
            v=random.choice(nodes) #v is chosen with a probability proportional to its degree
            G.add_edge(u,v)
            nodes.append(u)
            nodes.append(v)
        else:
            v=random.choice([x for x in range(n) if x!=u]) #any node different from u with the same probability
            G.add_edge(u,v)
            nodes.append(u)
            nodes.append(v)
    return G

# Generalized Watts-Strogatz (EK 20)
# n is the number of nodes (we assume n is a perfect square or it will be rounded to the closest perfect square)
# r is the radius of each node (a node u is connected with each other node at distance at most r) - strong ties
# k is the number of random edges for each node u - weak ties
#
# Here, the weak ties are still proportional to distance
# q is a term that evaluate how much the distance matters.
# Specifically, the probability of an edge between u and v is proportional to 1/dist(u,v)**q
# Hence, q=0 means that weak ties are placed uniformly at random, q=infinity only place weak ties towards neighbors.
#
# Next implementation of Watts-Strogatz graphs assumes that nodes are on a two-dimensional space (similar implementation can be given on larger dimensions).
# Here, distance between nodes will be set to be the Euclidean distance.
# This approach allows us a more fine-grained and realistic placing of nodes (i.e., they not need to be all at same distance as in the grid)
def GenWS2DG(n, r, k, q):
    G = nx.Graph()

    # We assume that the 2D area is sqrt(n) x sqrt(n) for sake of comparison with the grid implementation.
    # Anyway, one may consider a larger or a smaller area.
    # However, recall that the radius r given in input must be in the same order of magnitude as the size of the area
    # (e.g., you cannot consider the area as being a unit square, and consider a radius 2, otherwise there will be an edge between each pair of nodes)
    line=int(math.sqrt(n))
    nodes=dict() #This will be used to associate to each node its coordinates
    prob=dict() #Keeps for each pair of nodes (u,v) the term 1/dist(u,v)**q

    # The following for loop creates n nodes and place them randomly in the 2D area.
    # If one want to consider a different placement, e.g., for modeling communities, one only need to change this part.
    for i in range(n):
        x=random.random()
        y=random.random()
        nodes[i]=(x*line,y*line)
        prob[i]=dict()

    for i in range(n):
        # Strong-ties
        for j in range(i+1,n): #we add edge only towards next nodes, since edge to previous nodes have been added when these nodes have been processed
            dist=math.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2) #Euclidean Distance
            prob[i][j]=1/(dist**q)
            prob[j][i]=prob[i][j]
            if dist <= r:
                G.add_edge(i,j)

        # Terms 1/dist(u,v)**q are not probabilities since their sum can be different from 1.
        # To translate them in probabilities we normalize them, i.e, we divide each of them for their sum
        norm=sum(prob[i].values())
        # Weak ties
        # They are not exactly h, since the random choice can return a node s such that edge (i, s) already exists
        for h in range(k):
            # Next instruction allows to choice from the list given as first argument according to the probability distribution given as second argument
            s=numpy.random.choice([x for x in range(n) if x != i],p=[prob[i][x]/norm for x in range(n) if x!= i])
            G.add_edge(i,s)

    return G


# Affiliation Networks (Lattanzi & Sivakumar, STOC 2009)
# This merges the preferential attachment approach (that allows to model power law degree distributions)
# and the Watts-Strogatz approach (that allows to create smallworld networks).
# Moreover, it allows to easily model the community structure of networks.
# It works as follows: nodes do not decide to attach to other nodes, but they decide to affiliate to communities.
# This choice is done in a way that resembles the preferential affiliation model, i.e., nodes tend to choose communities with many nodes.
# Only once each node has been affiliated to communities, edge will be created.
# Strong ties are created among nodes within communities.
# Moreover, each node has also a number of weak ties. These are chosen according to a preferential attachment approach.
#
# n = number of nodes
# m = number of communities
# q = probability of preferential affiliation to communities
# c = maximum number of communities to which one node may be affiliated
# p = probability of an inter-community edge (strong ties)
# s = number of out-community edges (weak ties)
def affiliationG(n, m, q, c, p, s):
    G = nx.Graph()
    community=dict() #It keeps for each community the nodes that it contains
    for i in range(m):
        community[i]=set()
    comm_inv=dict() #It keeps for each node the communities to which is affiliated
    for i in range(n):
        comm_inv[i]=set()
    # Keeps each node as many times as the number of communities in which is contained
    # It serves for the preferential affiliation to communities (view below)
    communities=[]
    #Keeps each node as many times as its degree
    #It serves for the preferential attachment of weak ties (view below)
    nodes=[]

    for i in range(n):
        # Preferential Affiliation to communities
        r=random.random()
        # Preferential Affiliation is done only with probability q (view else block).
        # With remaining probability (or if i is the first node to be processed and thus preferential attachment is not possible),
        # i is affiliated to at most c randomly chosen communities
        if len(communities) == 0 or r > q:
            num_com=random.randint(1,c) #number of communities is chosen at random among 1 and c
            for k in range(num_com):
                comm=random.choice([x for x in range(m)])
                community[comm].add(i)
                if comm not in comm_inv[i]:
                    comm_inv[i].add(comm)
                    communities.append(i)
        else:
            #Here, we make preferential affiliation: a node is chosen proportionally to the number of communities in which it is contained
            #and is copied (i.e., i is affilated to the same communities containing the chosen node).
            #Observe that the probability that i is affilated to a given community increases when the number of nodes in that community is large,
            #since the probability of selecting a node from a large community is larger than from a small community
            prot=random.choice(communities) #Choose a prototype to copy
            for comm in comm_inv[prot]:
                community[comm].add(i)
                if comm not in comm_inv[i]:
                    comm_inv[i].add(comm)
                    communities.append(i)

        # Strong ties (edge within communities)
        # For each community and each node within that community we add an edge with probability p
        for comm in comm_inv[i]:
            for j in community[comm]:
                if j != i and not G.has_edge(i,j):
                    r=random.random()
                    if r <= p:
                        G.add_edge(i,j)
                        nodes.append(i)
                        nodes.append(j)

        # Preferential Attachment of weak ties
        # We choose s nodes with a probability that is proportional to their degree and we add an edge to these nodes
        if len(nodes) == 0:  #if i is the first node to be processed (and thus preferential attachment is impossible), then the s neighbors are selected at random
            for k in range(s):
                v = random.choice([x for x in range(n) if x!=i])
                if not G.has_edge(i,v):
                    G.add_edge(i,v)
                    nodes.append(i)
                    nodes.append(v)
        else:
            for k in range(s):
                v = random.choice(nodes)
                if not G.has_edge(i,v):
                    G.add_edge(i,v)
                    nodes.append(i)
                    nodes.append(v)

    return G

# print(randomG(9,0.5).edges())
# print(configurationG([5,3,3,2,2,1,1,1,1]).edges())
# deg_list=power_law_degree(9,2)
# print(deg_list)
# print(configurationG(deg_list).edges())
# print(preferentialG(9,0.75).edges())
# print(GenWS2DG(9, 1, 1, 2).edges())
# print(affiliationG(9, 4, 0.5, 3, 0.8, 2).edges())
