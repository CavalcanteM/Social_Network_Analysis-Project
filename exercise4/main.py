import csv
import math
from sklearn.linear_model import LinearRegression
import time
from dataset_generator import read_dataset_HC, generate_alt_dataset_HC, read_dataset_ICLR, generate_alt_dataset_ICLR
from hill_climbing import HillClimbing
from iclr import ICLogisticRegression
import random

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

# Loading the standard dataset, i.e. the one with all the 288 cases
filename = "../training.csv"
data_two_feature, labels_two_feature, data_three_feature, labels_three_feature, \
    test, labels_test = read_dataset_HC(filename)

# Training the classifiers on the standard dataset
start = time.time()
two_feature = LinearRegression().fit(data_two_feature, labels_two_feature)
three_feature = LinearRegression().fit(data_three_feature, labels_three_feature)
end = time.time()
print("Training time: ", end - start)

# Classification of the samples in the test set 
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
print("Classification of the test set time: ", end - start)
print("Wrong: ", wrong)
print("Correct: ", correct)

data_two_feature_altered = []
labels_two_feature_altered = []
data_three_feature_altered = []
labels_three_feature_altered = []

# Generation of the altered dataset starting from the standard one
data_two_feature_altered, labels_two_feature_altered, data_three_feature_altered, \
    labels_three_feature_altered = generate_alt_dataset_HC(data_two_feature, labels_two_feature, data_three_feature, labels_three_feature, 100, 500)

# Training the classifiers on the altered dataset
start = time.time()
two_feature_altered = LinearRegression().fit(data_two_feature_altered, labels_two_feature_altered)
three_feature_altered = LinearRegression().fit(data_three_feature_altered, labels_three_feature_altered)
end = time.time()
print("Training time (altered version): ", end - start)

# Classification of the samples in the test set on the altered version
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
print("Classification of the test set time (altered version): ", end - start)
print("Wrong: ", wrong)
print("Correct: ", correct)

# Truthfulness verification
data_with_star = []
data_with_zero = []
# Loading all the 288 possible cases and selecting the ones that have a '*' in service or value features, 
# and generating the corrisponding sample with a 0 on that feature
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
    # If the sample with a '*' receives a scores greater than the one with a value on that feature, the classifier is not truthful
    if HillClimbing(el1,two_feature,three_feature) > HillClimbing(el2,two_feature,three_feature):
        truthful = 0
        break

print("Truthful: ", truthful == 1)
# Truthfulness for altered dataset classifier
truthful = 1
for el1, el2 in zip(data_with_star, data_with_zero):
    # If the sample with a '*' receives a scores greater than the one with a value on that feature, the classifier is not truthful
    if HillClimbing(el1,two_feature_altered,three_feature_altered) > HillClimbing(el2,two_feature_altered,three_feature_altered):
        print(el1)
        print(HillClimbing(el1,two_feature_altered,three_feature_altered))
        print(el2)
        print(HillClimbing(el2,two_feature_altered,three_feature_altered))
        truthful = 0
        break

print("Truthful (altered): ", truthful == 1)

######################################################################
############   Incentive Compatible Logistic Regression   ############
######################################################################

print("\n\n############   Incentive Compatible Logistic Regression   ############\n")

# Loading the standard dataset, i.e. the one with all the 288 cases, and generating the altered version
data, test = read_dataset_ICLR(filename)
data = generate_alt_dataset_ICLR(data, 100, 500)

# Generating the dataset for the first LR, i.e. assigns label = 0 to samples with 1 star, label = 1 to samples with 2 or 3 stars
data1 = []
for row in data:
    if row[-1] == 1:
        label = 0
    else:
        label = 1
    data1.append([int(row[0]), int(row[1]), int(row[2]), label])

# Generating the dataset for the first LR, i.e. assigns label = 0 to 2-star samples, label = 1 to 3-star samples, thus excluding all 1-star samples
data2 = []
for row in data:
    if row[-1] == 1:
        pass
    elif row[-1] == 2:
        label = 0
    else:
        label = 1
    data2.append([int(row[0]), int(row[1]), int(row[2]), label])

# First Logistic Regression training
start = time.time()
lr = 10**-4
delta = 10**-5 
beta = [-random.random(),random.random(),random.random(),random.random()] #beta is selected randomly 
parameters = ICLogisticRegression(data1, lr, delta, beta)
end = time.time()
print("Training time first classifier: ", end - start)

start = time.time()
# Second Logistic Regression training
beta = [-random.random(),random.random(),random.random(),random.random()] #beta is selected randomly 
parameters1 = ICLogisticRegression(data2, lr, delta, beta)
end = time.time()
print("Training time second classifier: ", end - start)


# Classification of the samples in the test set
start = time.time()
right = 0
wrong = 0
for item in test:
    k = parameters[0] + item[0]*parameters[1] + item[1]*parameters[2] + item[2]*parameters[3]
    value = (1 / (1 + math.exp(-k)))
    if value >= 0.5: # the sample has a score of 2 o 3 stars
        k1 = parameters1[0] + item[0]*parameters1[1] + item[1]*parameters1[2] + item[2]*parameters1[3]
        value1 = (1 / (1 + math.exp(-k1)))
        if value1 >= 0.5: # the sample has a score of 3 stars
            pred_label = 3
        else: # the sample has a score of 2 stars
            pred_label = 2
    else: # the sample has a score of 1 stars
        pred_label = 1
    # Check if the predicted label is different from the baseline in the ground truth
    if pred_label == item[3]:
        right += 1
    else:
        wrong += 1

end = time.time()
print("Classification of the test set time: ", end - start)
print("Wrong: ", wrong)
print("Correct: ", correct)

# Truthfulness 
truthful = 1
for el_with_star, el_with_zero in zip(data_with_star,data_with_zero):
    # We have to substitute every '*' values with -1
    new_el_with_star = []
    for i in el_with_star:
        if i == '*':
            new_el_with_star.append(-1)
        else:
            new_el_with_star.append(int(i))
    # Compute k value for the element with a '*'
    k_with_star = parameters[0] + int(new_el_with_star[0])*parameters[1] + int(new_el_with_star[1])*parameters[2] + int(new_el_with_star[2])*parameters[3]
    # Compute k value for the element without a '*'
    k_with_zero = parameters[0] + int(el_with_zero[0])*parameters[1] + int(el_with_zero[1])*parameters[2] + int(el_with_zero[2])*parameters[3]
    # Compute value for the element with a '*'
    value_with_star = (1 / (1 + math.exp(-k_with_star)))
    # Compute value for the element without a '*'
    value_with_zero = (1 / (1 + math.exp(-k_with_zero)))
    # Classifign the element with a star
    if value_with_star >= 0.5: # the sample with '*' has a score of 2 o 3 stars
        k1 = parameters1[0] + int(new_el_with_star[0])*parameters1[1] + int(new_el_with_star[1])*parameters1[2] + int(new_el_with_star[2])*parameters1[3]
        value1 = (1 / (1 + math.exp(-k1)))
        if value1 >= 0.5: # the sample with '*' has a score of 3 stars
            pred_label_with_star = 3
        else: # the sample with '*' has a score of 2 stars
            pred_label_with_star = 2
    else: # # the sample with '*' has a score of 1 star
        pred_label_with_star = 1
    # Classifign the element without a star
    if value_with_zero >= 0.5: # the sample without '*' has a score of 2 o 3 stars
        k1 = parameters1[0] + int(el_with_zero[0])*parameters1[1] + int(el_with_zero[1])*parameters1[2] + int(el_with_zero[2])*parameters1[3]
        value1 = (1 / (1 + math.exp(-k1)))
        if value1 >= 0.5: # the sample without '*' has a score of 3 stars
            pred_label_with_zero = 3
        else: # the sample without '*' has a score of 2 stars
            pred_label_with_zero = 2
    else: # the sample without '*' has a score of 1 star
        pred_label_with_zero = 1
    # If the sample with a '*' receives a scores greater than the one with a value on that feature, the classifier is not truthful
    if pred_label_with_star > pred_label_with_zero:
        truthful = 0

print("LR Truthful: ", truthful == 1)