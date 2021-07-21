import networkx as nx
import numpy as np
from selection import selector


# p is the vector that contains the politic orientation of each candidate
# b is the vector that contains the politic orientation of each voter
def plurality_voting_rule(p, b):
    votes = {i: 0 for i in range(len(p))}   # Dict that contains the number of votes for each candidate

    # For each voter's politic orientation we check the minimum distance candidate
    for i in range(len(b)):
        min_index = 0          # Set the first candidate as the initial minimum distance candidate
        min_difference = abs(p[0] - b[i])   # Save the distance between the voter i and candidate 0
        # We save the sign, because in case of draw, win the the leftmost candidate
        sign = np.sign(p[0] - b[i])
        for x in range(1, len(p)):  # Start from 1, because candidate 0 is the default
            diff = abs(p[x] - b[i])
            if diff < min_difference or (diff == min_difference and sign == 1):
                # The candidate x is the actual minimum distance candidate
                min_index = x
                min_difference = diff
                sign = np.sign(p[x] - b[i])
        # The candidate min_index receveid a vote from the voter i
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
                if opinions[i] - prev_opinions[i] < 10 ** -5:
                    stop += 1

        if stop < G.number_of_nodes():
            stop = 0
        prev_opinions = opinions.copy()
        t += 1  # Update time step

    return opinions


# G -> graph
# p -> list of the politic orientation of each candidate
# c -> index of the selected candidate
# B -> number of seeds to select
# b -> vector of initial belief
def manipulation(G, p, c, B, b):
    votes = plurality_voting_rule(p, b)
    # SELECTION OF B NODES
    seeds = selector(G, B)

    # Stubborness definition
    stubborness = np.ones(G.number_of_nodes())/2

    for x in seeds:
        stubborness[x] = 1
        b[x] = p[c]

    new_b = FriedkinJohnsen(G, stubborness, b)

    new_votes = plurality_voting_rule(p, new_b)

    print("13," + str(votes[c]) + "," + str(new_votes[c]))
