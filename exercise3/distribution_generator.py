import csv
import random

# For the first distribution we assume that the node s represents the label 1 and the node t represents the
# label 2 and 3.
# We assume that the values have a probability D+(x) related how much is high the mean of the three score in respect
# with a threshold value defined to be 2 for the cut 1-23 and 4 for the cut 2-3
def dataset_generator(min_samples, max_samples):
    with open('../training.csv', 'r') as f:
        data = csv.reader(f, delimiter=',')

        with open('dataset_distribution1.csv', 'w', newline='\n') as f1:
            new_data1 = csv.writer(f1, delimiter=',')

            with open('dataset_distribution2.csv', 'w', newline='\n') as f2:
                new_data2 = csv.writer(f2, delimiter=',')

                with open('dataset_distribution3.csv', 'w', newline='\n') as f3:
                    new_data3 = csv.writer(f3, delimiter=',')
                    for item in data:
                        if item[1] == '*':
                            mean = (int(item[0]) + int(item[2]))/3
                        elif item[2] == '*':
                            mean = (int(item[0]) + int(item[1])) / 3
                        else:
                            mean = (int(item[0]) + int(item[1]) + int(item[2])) / 3

                        # probability for the first distribution  
                        if mean >= 3:
                            p0 = 1/2 + (mean - 3)/4 # /4 because the max difference is 2, so with 5 we have 1
                        else:
                            p0 = 1/2 + (mean - 3)/6 # /6 because if we have a number ->3, we must have -> 1/2

                        # probability for the second distribution
                        if mean >= 2:
                            p1 = 1/2 + (mean-2)/6   # /6 because the max difference is 3, so with 5 we have 1
                        else:
                            p1 = 1/2 + (mean-2)/4    # /4 because if we have a number ->2, we must have -> 1/2

                        # probability for the third distribution
                        if mean >= 4:
                            p2 = 1/2 + (mean-4)/2  # /6 because the max difference is 3, so with 5 we have 1
                        elif mean >= 2:
                            p2 = 1/2 + (mean-4)/4  # /4 because if we have a number ->2, we must have -> 1/2
                        else:
                            p2 = 0

                        for _ in range(random.choice(range(min_samples,max_samples+1))):
                            r = random.random()

                            if r < p0:
                                new_item = [item[0], item[1], item[2], '1']
                            else:
                                new_item = [item[0], item[1], item[2], '0']
                            new_data1.writerow(new_item)

                            if r < p1:
                                new_item = [item[0], item[1], item[2], '1']
                            else:
                                new_item = [item[0], item[1], item[2], '0']
                            new_data2.writerow(new_item)

                            if r < p2:
                                new_item = [item[0], item[1], item[2], '1']
                            else:
                                new_item = [item[0], item[1], item[2], '0']
                            new_data3.writerow(new_item)


def distribution():
    with open('dataset_distribution1.csv', 'r') as f0:
        data0 = csv.reader(f0, delimiter=',')

        dist0 = {}
        num0 = {}
        for item in data0:
            vote = (item[0], item[1], item[2])
            if vote in dist0.keys():
                dist0[vote] += int(item[3])
                num0[vote] += 1
            else:
                dist0[vote] = int(item[3])
                num0[vote] = 1
        for k in dist0.keys():
            dist0[k] = dist0[k]/num0[k]

    with open('dataset_distribution2.csv', 'r') as f1:
        data1 = csv.reader(f1, delimiter=',')

        dist1 = {}
        num1 = {}
        for item in data1:
            vote = (item[0], item[1], item[2])
            if vote in dist1.keys():
                dist1[vote] += int(item[3])
                num1[vote] += 1
            else:
                dist1[vote] = int(item[3])
                num1[vote] = 1
        for k in dist1.keys():
            dist1[k] = dist1[k]/num1[k]

    with open('dataset_distribution3.csv', 'r') as f2:
        data2 = csv.reader(f2, delimiter=',')
        dist2 = {}
        num2 = {}
        for item in data2:
            vote = (item[0], item[1], item[2])
            if vote in dist2.keys():
                dist2[vote] += int(item[3])
                num2[vote] += 1
            else:
                dist2[vote] = int(item[3])
                num2[vote] = 1
        for k in dist2.keys():
            dist2[k] = dist2[k] / num2[k]

    return dist0, dist1, dist2
