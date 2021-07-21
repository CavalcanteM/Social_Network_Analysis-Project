import csv
import random
import numpy as np

def read_dataset_HC(filename): # Generates the dataset required for HC
    # random.seed(42)
    data_two_feature = [] # Array that contains all the training samples with only two feature, i.e. presenting a '*' in service or value
    labels_two_feature = [] # Array that contains the labels of the ground truth for each training sample with two features
    data_three_feature = [] # Array that contains all the training samples with three feature, i.e. not presenting a '*' in service or value
    labels_three_feature = [] # Array that contains the labels of the ground truth for each training sample with three features
    test = [] # Array that contains all the test samples
    labels_test = [] # Array that contains the labels of the ground truth for each test sample

    # The dataset is splitted in 80% training and 20% test sets
    with open(filename, "r") as f:
        rows = csv.reader(f, delimiter=",")
        for row in rows:
            if random.random() < 0.8:
                new_row = []
                label = []
                for el in row[:-1]:
                    if el != '*': # Values '*' are not included in the training set
                        new_row.append(int(el))
                label = row[-1]
                if len(new_row) == 2: # separating samples with two features from ones with three features
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

# Generates an altered version of the datasets for HC
def generate_alt_dataset_HC(data_two_feature, labels_two_feature, data_three_feature, labels_three_feature, min_samples, max_samples):
    # random.seed(42)
    data_two_feature_altered = [] # Array that contains all the altered training samples with only two feature, i.e. presenting a '*' in service or value
    labels_two_feature_altered = [] # Array that contains the altered labels of the ground truth for each training sample with two features
    data_three_feature_altered = [] # Array that contains all the altered training samples with three feature, i.e. not presenting a '*' in service or value
    labels_three_feature_altered = [] # Array that contains the altered labels of the ground truth for each training sample with three features

    for el, label in zip(data_two_feature, labels_two_feature): # For each samples with two feature and its corrisponding label
        value = 0
        for i in el:
            value += i
        value = value / 2 # Computing the mean of the values of the samples
        for i in range(random.randrange(min_samples, max_samples + 1)): # Generating a random number n of new samples from the basic one, min_samples <= n <= max_samples
            # in which the label associated to it is altered
            data_two_feature_altered.append(el) # Adding the sample
            if value < 1.5: # if mean is less than 1.5, the original samples has a score of 1 star and is far from the threshold of 2
                p = random.choice(np.arange(0.03,0.08,0.01)) # possibility to became altered, 3% < p < 8%
            elif value >= 1.5 and value < 2.5: # if 1.5 <= mean < 2.5, the original samples has a score of 1 or 2 stars and is near the the threshold of 2
                p = random.choice(np.arange(0.15,0.25,0.01)) # possibility to became altered, 15% < p < 25%
            elif value >= 2.5 and value < 3.5: # if 2.5 <= mean < 3.5, the original samples has a score of 2 stars and is far from both the threshold of 2 and 4
                p = random.choice(np.arange(0.03,0.08,0.01)) # possibility to became altered, 3% < p < 8%
            elif value >= 3.5 and value < 4.5: # if 3.5 <= mean < 4.5, the original samples has a score of 2 or 3 stars and is near the threshold of 4
                p = random.choice(np.arange(0.15,0.25,0.01)) # possibility to became altered, 15% < p < 25%
            else: # mean is greater than 4.5, the original samples has a score of 3 stars and is far from the threshold of 4
                p = random.choice(np.arange(0.03,0.08,0.01)) # possibility to became altered, 3% < p < 8%
            if random.random() < (1-p): # the samples is not altered
                labels_two_feature_altered.append(label)
            else:
                if value < 1.5 or value > 4.5: # if the original label is 1 or 3, we assign 2 as new label
                    labels_two_feature_altered.append('2')
                else:
                    if value >= 1.5 and value < 2.5:
                        labels_two_feature_altered.append(str(random.choice([1,2]))) # the altered label can be both 1 or 2
                    elif value >= 2.5 and value < 3.5:
                        labels_two_feature_altered.append(str(random.choice([1,3]))) # the altered label can be both 1 or 3
                    else:
                        labels_two_feature_altered.append(str(random.choice([2,3]))) # the altered label can be both 2 or 3
                    

    # The same is repeated for samples with three features
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
                if value < 1.5 or value > 4.5:
                    labels_three_feature_altered.append('2')
                else:
                    if value >= 1.5 and value < 2.5:
                        labels_three_feature_altered.append(str(random.choice([1,2]))) # the altered label can be both 1 or 2
                    elif value >= 2.5 and value < 3.5:
                        labels_three_feature_altered.append(str(random.choice([1,3]))) # the altered label can be both 1 or 3
                    else:
                        labels_three_feature_altered.append(str(random.choice([2,3]))) # the altered label can be both 2 or 3

    return data_two_feature_altered, labels_two_feature_altered, data_three_feature_altered, labels_three_feature_altered

# Generates the dataset required for ICLR
def read_dataset_ICLR(filename):
    random.seed(42)
    data = [] # Array that contains all the training samples and their labels
    test = [] # Array that contains all the test samples and their labels

    with open(filename, "r") as f:
        rows = csv.reader(f, delimiter=",")
        for row in rows:
            new_row = []
            for el in row[:-1]:
                # Substitute the '*' values with -1, in order to differentiate them from samples that have a score of 0 or more on the same feature.
                if el != '*':
                    new_row.append(int(el))
                else:
                    new_row.append(-1)
            new_row.append(int(row[-1]))
            if random.random() < 0.8: # The dataset is splitted in 80% training and 20% test sets
                data.append(new_row)
            else:
                test.append(new_row)

    return data, test

# Generates an altered version of the datasets for ICLR, works similar to the version for HC
def generate_alt_dataset_ICLR(data, min_samples, max_samples):
    random.seed(42)
    data_altered = []

    for el in data:
        value = 0
        for i in el[:-1]:
            value += i
        value = value / 2
        for i in range(random.randrange(min_samples, max_samples + 1)):
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
                data_altered.append(el)
            else:
                if value < 1.5 or value > 4.5:
                    data_altered.append([el[0],el[1],el[2],2])
                else:
                    if value >= 1.5 and value < 2.5:
                        data_altered.append([el[0],el[1],el[2],random.choice([1,2])]) # the altered label can be both 1 or 2
                    elif value >= 2.5 and value < 3.5:
                        data_altered.append([el[0],el[1],el[2],random.choice([1,3])]) # the altered label can be both 1 or 3
                    else:
                        data_altered.append([el[0],el[1],el[2],random.choice([2,3])]) # the altered label can be both 2 or 3
                    
    return data_altered