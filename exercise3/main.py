from distribution_generator import dataset_generator, distribution
import networkx as nx
import csv
from mincut import MinCut
import time

start = time.time()
dataset_generator() 
d1, d2 = distribution()
end = time.time()
print("Tempo generazione distribuzione: ", end - start)

start = time.time()
one_star_partition, two_star_partition, three_star_partition = MinCut(d1, d2)
end = time.time()
print("Tempo creazione partizioni: ", end - start)

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

start = time.time()
data_with_zero = []
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

truthful = 1
for item in data_with_zero:
    if (item[0], item[1], item[2]) in one_star_partition:
        print(item)
        truthful = 0
        break

print("Truthful altered: ", truthful)
end = time.time()
print("Tempo verifica truthfulness: ", end - start)
