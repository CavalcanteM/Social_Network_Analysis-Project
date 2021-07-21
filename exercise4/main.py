import csv
import math
from sklearn.linear_model import LinearRegression
import time
from dataset_generator import read_dataset, generate_alt_dataset
from hill_climbing import HillClimbing
from iclr import ICLogisticRegression


###########################################
############   Hill Climbing   ############
###########################################

print("############   Hill Climbing   ############\n")

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


######################################################################
############   Incentive Compatible Logistic Regression   ############
######################################################################

print("\n\n############   Incentive Compatible Logistic Regression   ############\n")

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
beta = [-5, 0.6, 0.4, 0.3] # [b0, b1, b2, b3]
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
    if value >= 0.5: # 2 o 3 stars
        k1 = parameters1[0] + item[0]*parameters1[1] + item[1]*parameters1[2] + item[2]*parameters1[3]
        value1 = (1 / (1 + math.exp(-k1)))
        if value1 >= 0.5: # 3 stars
            pred_label = 3
        else: # 2 stars
            pred_label = 2
    else: # 1 star
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
    if value_with_star >= 0.5: # 2 o 3 stars
        k1 = parameters1[0] + int(new_el_with_star[0])*parameters1[1] + int(new_el_with_star[1])*parameters1[2] + int(new_el_with_star[2])*parameters1[3]
        value1 = (1 / (1 + math.exp(-k1)))
        if value1 >= 0.5: # 3 stars
            pred_label_with_star = 3
        else: # 2 stars
            pred_label_with_star = 2
    else: # 1 star
        pred_label_with_star = 1
    if value_with_zero >= 0.5: # 2 o 3 stars
        k1 = parameters1[0] + int(el_with_zero[0])*parameters1[1] + int(el_with_zero[1])*parameters1[2] + int(el_with_zero[2])*parameters1[3]
        value1 = (1 / (1 + math.exp(-k1)))
        if value1 >= 0.5: # 3 stars
            pred_label_with_zero = 3
        else: # 2 stars
            pred_label_with_zero = 2
    else: # 1 star
        pred_label_with_zero = 1
    if pred_label_with_star > pred_label_with_zero:
        truthful = 0

print("LR Truthful: ", truthful)