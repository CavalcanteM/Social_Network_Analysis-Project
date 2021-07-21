import numpy as np

# Hill Climbing takes in input the sample to classify, and the pretrained classifiers on two features and three features
def HillClimbing(sample, two_feature_classifier, three_feature_classifier):
    new_row = []
    # Selecting just the values different from '*'
    for el in sample:
        if el != '*':
            new_row.append(int(el))
    if len(new_row) == 2: # If the sample has two feature
        return int(np.round(two_feature_classifier.predict([new_row]))) # Returns the predicted score, rounded to the nearest integer
    else:
        row1 = [new_row[0], new_row[1]] # Generating the samples with only food and service
        row2 = [new_row[0], new_row[2]] # Generating the samples with only food and value

        # Predicting the score of all the three samples
        row1_value = int(np.round(two_feature_classifier.predict([row1])))
        row2_value = int(np.round(two_feature_classifier.predict([row2])))
        row3_value = int(np.round(three_feature_classifier.predict([new_row])))

        return max(row1_value, row2_value, row3_value) # Returning the max of the three predicted scores.