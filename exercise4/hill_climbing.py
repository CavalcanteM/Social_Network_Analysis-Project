from sklearn.linear_model import LinearRegression
import csv
import numpy as np
import random
import time

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

random.seed(42)

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

# print("Experiment normal")
# print("Num element:",len(test))
# print("Right:",num_right)
# print("Wrong:",num_wrong)
# print("**************************")

data_two_feature_altered = []
labels_two_feature_altered = []
data_three_feature_altered = []
labels_three_feature_altered = []

for el, label in zip(data_two_feature, labels_two_feature):
    for i in range(random.randrange(10,21)):
        data_two_feature_altered.append(el)
        if random.random() < 0.8:
            labels_two_feature_altered.append(label)
        else:
            if label == '1' or label == '3':
                labels_two_feature_altered.append('2')
            else:
                c = random.choice([-1,1])
                labels_two_feature_altered.append(str(int(label)+c))

for el, label in zip(data_three_feature, labels_three_feature):
    for i in range(random.randrange(10000,20000)):
        data_three_feature_altered.append(el)
        if random.random() < 0.7:
            labels_three_feature_altered.append(label)
        else:
            if label == '1' or label == '3':
                labels_three_feature_altered.append('2')
            else:
                c = random.choice([-1,1])
                labels_three_feature_altered.append(str(int(label)+c))


t1 = time.time()
two_feature_altered = LinearRegression().fit(data_two_feature_altered, labels_two_feature_altered)
three_feature_altered = LinearRegression().fit(data_three_feature_altered, labels_three_feature_altered)
t2 = time.time()
print("Altered training time: ", t2-t1)

num_right = 0
num_wrong = 0
wrong_cases = []
t = 0
for i in range(len(test)):
    t1 = time.time()
    predicted = HillClimbing(test[i], two_feature_altered, three_feature_altered)
    t2 = time.time()
    t += (t2-t1)
    if predicted == int(labels_test[i]):
        num_right += 1
    else:
        wrong_cases.append([test[i],labels_test[i],predicted])
        num_wrong += 1

# print("Experiment altered completed")
# print("Medium time: ", t/len(test))
# print("Num element:",len(test))
# print("Right:",num_right)
# print("Wrong:",num_wrong)
# if num_wrong > 0:
#     print("Wrong cases:\n")
#     for el in wrong_cases:
#         print(el)
#         row = []
#         for i in el[0]:
#             if i == '*':
#                 row.append(0)
#             else:
#                 row.append(int(i))
#         print(HillClimbing(row, two_feature_altered, three_feature_altered))
#         print("\n")
# print("**************************")

data_with_star = []
data_with_zero = []
with open("training.csv", "r") as f:
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


truthful = 1
for el1, el2 in zip(data_with_star, data_with_zero):
    if HillClimbing(el1,two_feature,three_feature) > HillClimbing(el2,two_feature,three_feature):
        print(el1)
        print(HillClimbing(el1,two_feature,three_feature))
        print(el2)
        print(HillClimbing(el2,two_feature,three_feature))
        truthful = 0

print("Truthful normal: ", truthful)

truthful = 1
for el1, el2 in zip(data_with_star, data_with_zero):
    if HillClimbing(el1,two_feature_altered,three_feature_altered) > HillClimbing(el2,two_feature_altered,three_feature_altered):
        print(el1)
        print(HillClimbing(el1,two_feature_altered,three_feature_altered))
        print(el2)
        print(HillClimbing(el2,two_feature_altered,three_feature_altered))
        truthful = 0

print("Truthful altered: ", truthful)