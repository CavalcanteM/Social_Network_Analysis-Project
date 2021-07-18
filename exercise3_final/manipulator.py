import networkx as nx
import numpy as np
from selection import selector


# p è il vettore che indica l'orientamento politico dei candidati
# b è il vettore che indica l'orientamento politico dei votanti
def plurality_voting_rule(p, b):
    votes = {i: 0 for i in range(len(p))}   # Dizionario contenente il numero di voti per ogni candidato

    # Per ogni orientamento politico dei votanti andiamo a controllare il candidato a minima distanza
    for i in range(len(b)):
        min_index = 0          # Settiamo il primo candidato come quello a minima distanza
        min_difference = abs(p[0] - b[i])   # Settiamo la distanza tra il votante i e il candidato 0
        # Ci salviamo il segno in quanto, in caso di pareggio, dobbiamo scegliere il candidato a sinistra
        sign = np.sign(p[0] - b[i])
        for x in range(1, len(p)):  # Partiamo da 1 perché il candidato 0 lo abbiamo assegnato come min di default
            diff = abs(p[x] - b[i])
            if diff < min_difference or (diff == min_difference and sign == 1):
                # Il candidato x è quello a minima distanza
                min_index = x
                min_difference = diff
                sign = np.sign(p[x] - b[i])
        # Il candidato min_index ha ricevuto un voto dal votante i
        votes[min_index] += 1

    return votes


def FriedkinJohnsen(G, stubborness, belief):
    t = 0  # time step
    stop = 0  # stop condition
    opinions = []  # opinions at current time step
    prev_opinions = []  # opinions at previous time step

    while stop < G.number_of_nodes():
        if t == 0:
            for i in range(len(belief)):
                opinions.append(belief[i])
        else:
            for i in range(len(belief)):
                sum = 0
                for v in G.neighbors(str(i)):
                    sum += prev_opinions[int(v)] / G.degree(str(i))
                opinions[i] = stubborness[i] * belief[i] + (1 - stubborness[i]) * sum
                # if opinions[i] == prev_opinions[i]:
                if opinions[i] - prev_opinions[i] < 10 ** -5:
                    stop += 1

        if stop < G.number_of_nodes():
            stop = 0
        prev_opinions = opinions.copy()
        t += 1  # Update time step

    return opinions


# G -> grafo
# p -> lista dell'orientamento politico dei candidati
# c -> indice del candidato da favorire
# B -> numero di seeds da selezionare
# b -> il vettore delle belief iniziali
def manipulation(G, p, c, B, b):
    votes = plurality_voting_rule(p, b)
    b1 = b[:]
    b2 = b[:]
    b3 = b[:]
    b4 = b[:]
    b5 = b[:]
    b6 = b[:]
    b7 = b[:]
    b8 = b[:]
    # SELECTION OF B NODES
    seeds = selector(G, B)

    # Definizione della stubborness
    stubborness = np.ones(G.number_of_nodes())/2

    for x in seeds[0]:
        stubborness[x] = 1
        b1[x] = p[c]

    new_b1 = FriedkinJohnsen(G, stubborness, b1)

    new_votes1 = plurality_voting_rule(p, new_b1)

    print("SHAPLEY CLOSENESS," + str(votes[c]) + "," + str(new_votes1[c]))

    # Definizione della stubborness
    stubborness = np.ones(G.number_of_nodes()) / 2

    for x in seeds[1]:
        stubborness[x] = 1
        b2[x] = p[c]

    new_b2 = FriedkinJohnsen(G, stubborness, b2)

    new_votes2 = plurality_voting_rule(p, new_b2)

    print("SHAPLEY DEGREE," + str(votes[c]) + "," + str(new_votes2[c]))

    # Definizione della stubborness
    stubborness = np.ones(G.number_of_nodes()) / 2

    for x in seeds[2]:
        stubborness[x] = 1
        b3[x] = p[c]

    new_b3 = FriedkinJohnsen(G, stubborness, b3)

    new_votes3 = plurality_voting_rule(p, new_b3)

    print("SHAPLEY THRESHOLD," + str(votes[c]) + "," + str(new_votes3[c]))

    # Definizione della stubborness
    stubborness = np.ones(G.number_of_nodes()) / 2

    for x in seeds[3]:
        stubborness[x] = 1
        b4[x] = p[c]

    new_b4 = FriedkinJohnsen(G, stubborness, b4)

    new_votes4 = plurality_voting_rule(p, new_b4)

    print("CLOSENESS," + str(votes[c]) + "," + str(new_votes4[c]))

    # Definizione della stubborness
    stubborness = np.ones(G.number_of_nodes()) / 2

    for x in seeds[4]:
        stubborness[x] = 1
        b5[x] = p[c]

    new_b5 = FriedkinJohnsen(G, stubborness, b5)

    new_votes5 = plurality_voting_rule(p, new_b5)

    print("SHAPLEY CLOSENESS+DEGREE+THRESHOLD," + str(votes[c]) + "," + str(new_votes5[c]))

    # Definizione della stubborness
    stubborness = np.ones(G.number_of_nodes()) / 2

    for x in seeds[5]:
        stubborness[x] = 1
        b6[x] = p[c]

    new_b6 = FriedkinJohnsen(G, stubborness, b6)

    new_votes6 = plurality_voting_rule(p, new_b6)

    print("SHAPLEY CLOSENESS+DEGREE+THRESHOLD+DEGREE NORMALE," + str(votes[c]) + "," + str(new_votes6[c]))

    # Definizione della stubborness
    stubborness = np.ones(G.number_of_nodes()) / 2

    for x in seeds[6]:
        stubborness[x] = 1
        b7[x] = p[c]

    new_b7 = FriedkinJohnsen(G, stubborness, b7)

    new_votes7 = plurality_voting_rule(p, new_b7)

    print("SHAPLEY CLOSENESS+DEGREE+THRESHOLD+CLOSENESS NORMALE," + str(votes[c]) + "," + str(new_votes7[c]))

    # Definizione della stubborness
    stubborness = np.ones(G.number_of_nodes()) / 2

    for x in seeds[7]:
        stubborness[x] = 1
        b8[x] = p[c]

    new_b8 = FriedkinJohnsen(G, stubborness, b8)

    new_votes8 = plurality_voting_rule(p, new_b8)

    print("SHAPLEY CLOSENESS+DEGREE+THRESHOLD+(CLOSENESS e DEGREE NORMALE)," + str(votes[c]) + "," + str(new_votes8[c]))

    return [votes[c], new_votes1[c], new_votes2[c], new_votes3[c], new_votes4[c], new_votes5[c], new_votes6[c], new_votes7[c], new_votes8[c]]
