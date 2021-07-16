import csv
import random

# with open("training") as f:
#     rows = csv.reader(f, delimiter=" ")

#     train = []
#     p = 0.9
#     for row in rows:
#         if random.random() < p: # devo includere
#             num = int(random.uniform(5, 10)) # numero samples uguali
#             num_right = int(num * 4/5)
#             num_wrong = num - num_right

#             for i in range(num_right):
#                 train.append(row)

#             for i in range(num_wrong):
#                 if int(row[-1]) == 3:
#                     label = 2
#                 elif int(row[-1]) == 2:
#                     label = random.choice([1, 3])
#                 else:
#                     label = 2

#                 new_row = [row[0], row[1], row[2], label]
#                 train.append(new_row)

# random.shuffle(train)
# with open("training_exp1.csv", "w", newline="\n") as f:
#     writer = csv.writer(f, delimiter=",")
#     for i in train:
#         writer.writerow(i)

with open("training_exp1.csv") as f:
    rows = csv.reader(f, delimiter=",")

    new_rows = []
    for row in rows:
        if int(row[-1]) == 2 or int(row[-1]) == 3:
            row[-1] = 1
        else:
            row[-1] = 0
        new_rows.append(row)

    with open("training_exp1_first.csv", "w", newline="\n") as f:
        writer = csv.writer(f, delimiter=",")
        for i in new_rows:
            writer.writerow(i)

with open("training_exp1.csv") as f:
    rows = csv.reader(f, delimiter=",")

    new_rows = []
    for row in rows:
        if int(row[-1]) == 2: 
            row[-1] = 0
            new_rows.append(row)
        elif int(row[-1]) == 3:
            row[-1] = 1
            new_rows.append(row)

    with open("training_exp1_second.csv", "w", newline="\n") as f:
        writer = csv.writer(f, delimiter=",")
        for i in new_rows:
            writer.writerow(i)           