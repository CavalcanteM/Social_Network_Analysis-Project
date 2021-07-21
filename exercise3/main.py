from distribution_generator import dataset_generator, distribution
import csv
from mincut import MinCut
import time

# Distribution generator for min cut algorithm
MIN_SAMPLES = 100
MAX_SAMPLES = 500

start = time.time()
dataset_generator(MIN_SAMPLES, MAX_SAMPLES) 
d0, d1, d2 = distribution()
end = time.time()
print("Tempo generazione distribuzione: ", end - start)

# Min cut algorithm execution
start = time.time()
one_star_partition, two_star_partition, three_star_partition = MinCut(d0, d1, d2)
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
truthful = 1
for row in one_star_partition:
    if row[1] != '*' and row[2] != '*':
        new_row1 = (row[0], '*', row[2])
        if new_row1 not in one_star_partition:
            truthful = 0
            break
        new_row2 = (row[0], row[1], '*')
        if new_row2 not in one_star_partition:
            truthful = 0
            break

for row in two_star_partition:
    if row[1] != '*' and row[2] != '*':
        new_row1 = (row[0], '*', row[2])
        if new_row1 in three_star_partition:
            truthful = 0
            break
        new_row2 = (row[0], row[1], '*')
        if new_row2 in three_star_partition:
            truthful = 0
            break
end = time.time()
print("Truthful: ", truthful)
print("Tempo verifica truthfulness: ", end - start)