import random

# Friedkin Johnsen dynamics
# In this function belief and stubborness are defined as random
# numbers for all the nodes
def FriedkinJohnsen(G):
    belief = {i:random.random() for i in G.nodes()}
    stubborness = {i:random.random() for i in G.nodes()}
    t = 0 # time step
    stop = 0 # stop condition
    opinions = {} # opinions at current time step
    prev_opinions = {} # opinions at previous time step
    
    while stop < G.number_of_nodes():
        if t == 0:
            for i in G.nodes():
                opinions[i] = belief[i]
        else:
            for i in G.nodes():
                sum = 0
                for v in G.neighbors(i):
                    sum += prev_opinions[v] / G.degree(i)
                opinions[i] = stubborness[i] * belief[i] + (1 - stubborness[i]) * sum
                #if opinions[i] == prev_opinions[i]:
                if opinions[i] - prev_opinions[i] < 10**-5:
                    stop += 1
    
        str = "Time step %d: %d nodi stabili" % (t, stop)
        print(str)

        if stop < G.number_of_nodes():
            stop = 0
        prev_opinions = opinions.copy()
        t += 1 # Update time step

    return opinions
