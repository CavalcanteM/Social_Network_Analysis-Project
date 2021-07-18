from operator import ne
from sklearn.linear_model import LinearRegression
import csv
import numpy as np
import random

def HillClimbing(sample, two_feature_classifier, three_feature_classifier):
    new_row = []
    for el in sample:
        if el != '*':
            new_row.append(int(el))
    if len(new_row) == 2:
        return int(np.round(two_feature_classifier.predict([new_row])))
    else:
        row1 = [new_row[0], new_row[1]]
        row2 = [new_row[0], new_row[2]]

        row1_value = int(np.round(two_feature_classifier.predict([row1])))
        row2_value = int(np.round(two_feature_classifier.predict([row2])))
        row3_value = int(np.round(three_feature_classifier.predict([new_row])))

        return max(row1_value, row2_value, row3_value)
        

data_two_feature = []
labels_two_feature = []
data_three_feature = []
labels_three_feature = []
test = []
labels_test = []

with open("training.csv", "r") as f:
    rows = csv.reader(f, delimiter=",")
    for row in rows:
        if random.random() < 0.8:
            new_row = []
            label = []
            for el in row[:-1]:
                if el != '*':
                    new_row.append(int(el))
            label = row[-1]
            if len(new_row) == 2:
                data_two_feature.append(new_row)
                labels_two_feature.append(label)
            else:
                data_three_feature.append(new_row)
                labels_three_feature.append(label)
        else:
            test.append(row[:-1])
            labels_test.append(row[-1])

two_feature = LinearRegression().fit(data_two_feature, labels_two_feature)
three_feature = LinearRegression().fit(data_three_feature, labels_three_feature)

num_right = 0
num_wrong = 0
for i in range(len(test)):
    if HillClimbing(test[i], two_feature, three_feature) == int(labels_test[i]):
        num_right += 1
    else:
        num_wrong += 1
print("Num element:",len(test))
print("Right:",num_right)
print("Wrong:",num_wrong)