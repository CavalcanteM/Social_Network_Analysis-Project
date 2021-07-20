import csv
import random
import math
from sklearn.linear_model import LinearRegression
import time
import sys
sys.path.append("..")
from exercise3.dataset_generator import read_dataset, generate_alt_dataset
from hill_climbing import HillClimbing

data_two_feature = []
labels_two_feature = []
data_three_feature = []
labels_three_feature = []
test = []
labels_test = []

# Dataset standard
filename = "../training.csv"
data_two_feature, labels_two_feature, data_three_feature, labels_three_feature, \
    test, labels_test = read_dataset(filename)

# Addestramento classificatori con dataset standard
start = time.time()
two_feature = LinearRegression().fit(data_two_feature, labels_two_feature)
three_feature = LinearRegression().fit(data_three_feature, labels_three_feature)
end = time.time()
print("Tempo addestramento classificatori: ", end - start)

# Classificazione su test set
start = time.time()
correct = 0
wrong = 0
for sample, label in zip(test, labels_test):
    pred_label = HillClimbing(sample, two_feature, three_feature)

    if pred_label == int(label):
        correct += 1
    else:
        wrong += 1
        print(sample, label, pred_label)

end = time.time()
print("Tempo classificazione samples: ", end - start)
print("Wrong:", wrong)
print("Correct:", correct)

data_two_feature_altered = []
labels_two_feature_altered = []
data_three_feature_altered = []
labels_three_feature_altered = []

# Dataset alterato
data_two_feature_altered, labels_two_feature_altered, data_three_feature_altered, \
    labels_three_feature_altered = generate_alt_dataset(data_two_feature, labels_two_feature, data_three_feature, labels_three_feature, 10000, 20000)

# Addestramento classificatori con dataset alterato
start = time.time()
two_feature_altered = LinearRegression().fit(data_two_feature_altered, labels_two_feature_altered)
three_feature_altered = LinearRegression().fit(data_three_feature_altered, labels_three_feature_altered)
end = time.time()
print("Tempo addestramento classificatori: ", end - start)

# Classificazione su test set con classificatore alterato
start = time.time()
correct = 0
wrong = 0
for sample, label in zip(test, labels_test):
    pred_label = HillClimbing(sample, two_feature_altered, three_feature_altered)

    if pred_label == int(label):
        correct += 1
    else:
        wrong += 1
        print(sample, label, pred_label)

end = time.time()
print("Tempo classificazione samples: ", end - start)
print("Wrong:", wrong)
print("Correct:", correct)

# Truthfulness verification
data_with_star = []
data_with_zero = []
with open("../training.csv", "r") as f:
    rows = csv.reader(f, delimiter=",")
    for row in rows:
        star_found = 0
        new_row = []
        for el in row[:-1]:
            if el == '*':
                new_row.append(0)
                star_found = 1
            else:
                new_row.append(int(el))
        if star_found == 1:
            data_with_star.append(row[:-1])
            data_with_zero.append(new_row)

# Truthfulness for standard dataset classifier
truthful = 1
for el1, el2 in zip(data_with_star, data_with_zero):
    if HillClimbing(el1,two_feature,three_feature) > HillClimbing(el2,two_feature,three_feature):
        print(el1)
        print(HillClimbing(el1,two_feature,three_feature))
        print(el2)
        print(HillClimbing(el2,two_feature,three_feature))
        truthful = 0
        break

print("Truthful normal: ", truthful)

# Truthfulness for altered dataset classifier
truthful = 1
for el1, el2 in zip(data_with_star, data_with_zero):
    if HillClimbing(el1,two_feature_altered,three_feature_altered) > HillClimbing(el2,two_feature_altered,three_feature_altered):
        print(el1)
        print(HillClimbing(el1,two_feature_altered,three_feature_altered))
        print(el2)
        print(HillClimbing(el2,two_feature_altered,three_feature_altered))
        truthful = 0
        break

print("Truthful altered: ", truthful)