import numpy as np

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