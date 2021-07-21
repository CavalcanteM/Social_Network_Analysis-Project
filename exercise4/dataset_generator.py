import csv
import random
import numpy as np

def read_dataset(filename):
    data_two_feature = []
    labels_two_feature = []
    data_three_feature = []
    labels_three_feature = []
    test = []
    labels_test = []

    with open(filename, "r") as f:
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

    return data_two_feature, labels_two_feature, data_three_feature, labels_three_feature, \
        test, labels_test

def generate_alt_dataset(data_two_feature, labels_two_feature, data_three_feature, labels_three_feature, min_samples, max_samples):
    data_two_feature_altered = []
    labels_two_feature_altered = []
    data_three_feature_altered = []
    labels_three_feature_altered = []

    for el, label in zip(data_two_feature, labels_two_feature):
        value = 0
        for i in el:
            value += i
        value = value / 2
        for i in range(random.randrange(min_samples, max_samples + 1)):
            data_two_feature_altered.append(el)
            if value < 1.5:
                p = random.choice(np.arange(0.03,0.08,0.01))
            elif value >= 1.5 and value < 2.5:
                p = random.choice(np.arange(0.15,0.25,0.01))
            elif value >= 2.5 and value < 3.5:
                p = random.choice(np.arange(0.03,0.08,0.01))
            elif value >= 3.5 and value < 4.5:
                p = random.choice(np.arange(0.15,0.25,0.01))
            else:
                p = random.choice(np.arange(0.03,0.08,0.01))
            if random.random() < (1-p):
                labels_two_feature_altered.append(label)
            else:
                if label == '1' or label == '3':
                    labels_two_feature_altered.append('2')
                else:
                    c = random.choice([-1,1])
                    labels_two_feature_altered.append(str(int(label)+c))

    for el, label in zip(data_three_feature, labels_three_feature):
        value = 0
        for i in el:
            value += i
        value = value / 2
        for i in range(random.randrange(min_samples, max_samples + 1)):
            data_three_feature_altered.append(el)
            if value < 1.5:
                p = random.choice(np.arange(0.03,0.08,0.01))
            elif value >= 1.5 and value < 2.5:
                p = random.choice(np.arange(0.15,0.25,0.01))
            elif value >= 2.5 and value < 3.5:
                p = random.choice(np.arange(0.07,0.12,0.01))
            elif value >= 3.5 and value < 4.5:
                p = random.choice(np.arange(0.25,0.35,0.01))
            else:
                p = random.choice(np.arange(0.03,0.08,0.01))
            if random.random() < (1-p):
                labels_three_feature_altered.append(label)
            else:
                if label == '1' or label == '3':
                    labels_three_feature_altered.append('2')
                else:
                    c = random.choice([-1,1])
                    labels_three_feature_altered.append(str(int(label)+c))

    return data_two_feature_altered, labels_two_feature_altered, data_three_feature_altered, labels_three_feature_altered