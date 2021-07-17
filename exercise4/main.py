import csv
import random
import math
from tqdm import tqdm

def ICLogisticRegression(data, lr, delta):
    t = 0 # time step
    parameters = [0, 0, 0, 0] # [b0, b1, b2, b3]
    delta_g = float('inf')
    error = 0
    prev_loss = 0

    while delta_g > delta:
        g = [0 for _ in range(len(parameters))]

        for v in tqdm(range(len(data)), desc="Epoch " + str(t) + ": "):
        # for item in data:
            k = parameters[0] + data[v][0]*parameters[1] + data[v][1]*parameters[2] + data[v][2]*parameters[3]
            g[0] = g[0] + 1 / (1 + math.exp(-k)) - data[v][3]

            for i in range(1,len(parameters)):
                g[i] = g[i] + (1 / (1 + math.exp(-k)) - data[v][3]) * data[v][i]

        for i in range(len(parameters)):
            parameters[i] = max(parameters[i] - lr * g[i], 0)

        for item in data:
            value = (1 / (1 + math.exp(-k)))
            if value >= 0.5:
                pred_label = 1
            else:
                pred_label = 0
            error += abs(pred_label - item[3])
        
        loss = error/len(data)
        delta_g = loss - prev_loss
        loss = prev_loss
        error = 0
        t = t + 1

    return parameters

with open("training_exp1_first.csv") as f:
    rows = csv.reader(f, delimiter=",")

    data = []
    for row in rows:
        for i in range(len(row)):
            if row[i] == "*":
                row[i] = -1

        new_row = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
        data.append(new_row)

    lr = 10**-2
    delta = 10**-2

    print(ICLogisticRegression(data, lr, delta))