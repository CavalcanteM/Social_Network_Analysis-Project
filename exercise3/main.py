from distribution_generator import dataset_generator, distribution
from iclr import ICLogisticRegression
import networkx as nx
import csv
from mincut import MinCut
import time
import math
import random

# Distribution generator for min cut algorithm
start = time.time()
dataset_generator() 
d1, d2 = distribution()
end = time.time()
print("Tempo generazione distribuzione: ", end - start)

# Min cut algorithm execution
start = time.time()
one_star_partition, two_star_partition, three_star_partition = MinCut(d1, d2)
end = time.time()
print("Tempo creazione partizioni: ", end - start)

# Label prediction on test samples
start = time.time()
correct = 0
wrong = 0
with open('../training.csv', 'r') as f:
    data = csv.reader(f, delimiter=',')
    for item in data:
        x = (item[0], item[1], item[2])
        if x in one_star_partition:
            if item[3] == '1':
                correct += 1
            else:
                wrong += 1
                print(x, item[3], '1')
        elif x in two_star_partition:
            if item[3] == '2':
                correct += 1
            else:
                wrong += 1
                print(x, item[3], '2')
        else:
            if item[3] == '3':
                correct += 1
            else:
                wrong += 1
                print(x, item[3], '3')
print("Wrong:", wrong)
print("Correct:", correct)
end = time.time()
print("Tempo classificazione samples: ", end - start)

# Truthfulness verification
start = time.time()
data_with_zero = []
data_with_star = []
for row in two_star_partition:
    star_found = 0
    new_row = []
    for i in row:
        if i == "*":
            new_row.append("0")
            star_found = 1
        else:
            new_row.append(i)
    if star_found == 1:
        data_with_zero.append(new_row)
        data_with_star.append(row)

truthful = 1
for item in data_with_zero:
    if (item[0], item[1], item[2]) in one_star_partition:
        print(item)
        truthful = 0
        break

print("Truthful: ", truthful)
end = time.time()
print("Tempo verifica truthfulness: ", end - start)

# Incentive Compatible Logistic Regression
data = []
with open("../training_exp1_first.csv") as f:
    rows = csv.reader(f, delimiter=",")

    for row in rows:
        for i in range(len(row)):
            if row[i] == "*":
                row[i] = -1

        new_row = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
        data.append(new_row)

# First Logistic Regression training
start = time.time()
lr = 10**-4
delta = 10**-5
beta = [-5,0.5,0,0.1] # [b0, b1, b2, b3]
parameters = ICLogisticRegression(data, lr, delta, beta)
end = time.time()
print("Tempo addestramento primo classificatore: ", end - start)

data = []
with open("../training_exp1_second.csv") as f:
    rows = csv.reader(f, delimiter=",")

    for row in rows:
        for i in range(len(row)):
            if row[i] == "*":
                row[i] = -1

        new_row = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
        data.append(new_row)

start = time.time()
# Second Logistic Regression training
beta = [-5, 0.6, 0.4, 0.3] # [b0, b1, b2, b3] # @TODO: da determinare
parameters1 = ICLogisticRegression(data, lr, delta, beta)
end = time.time()
print("Tempo addestramento secondo classificatore: ", end - start)

samples = []
with open("../training.csv") as f:
    rows = csv.reader(f, delimiter=",")

    for row in rows:
        for i in range(len(row)):
            if row[i] == "*":
                row[i] = -1

        new_row = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
        samples.append(new_row)

start = time.time()
# Classification of test samples
right = 0
wrong = 0
for item in samples:
    k = parameters[0] + item[0]*parameters[1] + item[1]*parameters[2] + item[2]*parameters[3]
    value = (1 / (1 + math.exp(-k)))
    if value >= 0.5: # 2 o 3 stelle
        k1 = parameters1[0] + item[0]*parameters1[1] + item[1]*parameters1[2] + item[2]*parameters1[3]
        value1 = (1 / (1 + math.exp(-k1)))
        if value1 >= 0.5: # 3 stelle
            pred_label = 3
        else: # 2 stelle
            pred_label = 2
    else: # 1 stella
        pred_label = 1

    if pred_label == item[3]:
        right += 1
    else:
        wrong += 1
        print(item)

print("Samples classificati correttamente: ", right)
print("Samples non classificati correttamente: ", wrong)
end = time.time()
print("Tempo classificazione samples: ", end - start)

truthful = 1
for el_with_star, el_with_zero in zip(data_with_star,data_with_zero):
    new_el_with_star = []
    for i in el_with_star:
        if i == '*':
            new_el_with_star.append(-1)
        else:
            new_el_with_star.append(int(i))
    k_with_star = parameters[0] + int(new_el_with_star[0])*parameters[1] + int(new_el_with_star[1])*parameters[2] + int(new_el_with_star[2])*parameters[3]
    k_with_zero = parameters[0] + int(el_with_zero[0])*parameters[1] + int(el_with_zero[1])*parameters[2] + int(el_with_zero[2])*parameters[3]
    value_with_star = (1 / (1 + math.exp(-k_with_star)))
    value_with_zero = (1 / (1 + math.exp(-k_with_zero)))
    if value_with_star >= 0.5: # 2 o 3 stelle
        k1 = parameters1[0] + int(new_el_with_star[0])*parameters1[1] + int(new_el_with_star[1])*parameters1[2] + int(new_el_with_star[2])*parameters1[3]
        value1 = (1 / (1 + math.exp(-k1)))
        if value1 >= 0.5: # 3 stelle
            pred_label_with_star = 3
        else: # 2 stelle
            pred_label_with_star = 2
    else: # 1 stella
        pred_label_with_star = 1
    if value_with_zero >= 0.5: # 2 o 3 stelle
        k1 = parameters1[0] + int(el_with_zero[0])*parameters1[1] + int(el_with_zero[1])*parameters1[2] + int(el_with_zero[2])*parameters1[3]
        value1 = (1 / (1 + math.exp(-k1)))
        if value1 >= 0.5: # 3 stelle
            pred_label_with_zero = 3
        else: # 2 stelle
            pred_label_with_zero = 2
    else: # 1 stella
        pred_label_with_zero = 1
    if pred_label_with_star > pred_label_with_zero:
        truthful = 0

print("LR Truthful: ", truthful)