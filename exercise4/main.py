import csv
import random
import math

def ICLogisticRegression(data, lr, delta):
    t = 0 # time step
    parameters = [-5,0.5,0,0.1] # [b0, b1, b2, b3]
    delta_g = float('inf')
    error = 0
    prev_loss = 0

    while delta_g > delta:
        g = [0 for _ in range(len(parameters))]

        for item in data:
            k = parameters[0] + item[0]*parameters[1] + item[1]*parameters[2] + item[2]*parameters[3]
            g[0] = g[0] + (1 / (1 + math.exp(-k))) - item[3]

            for i in range(1,len(parameters)):
                g[i] = g[i] + (1 / (1 + math.exp(-k)) - item[3]) * item[i]

        parameters[0] = parameters[0] - lr * g[0]
        for i in range(1, len(parameters)):
            parameters[i] = max(parameters[i] - lr * g[i], 0)

        for item in data:
            k = parameters[0] + item[0]*parameters[1] + item[1]*parameters[2] + item[2]*parameters[3]
            value = (1 / (1 + math.exp(-k)))
            if value >= 0.5:
                pred_label = 1
            else:
                pred_label = 0
            error += abs(pred_label - item[3])
        
        loss = error/len(data)
        print("LOSS " + str(loss))
        delta_g = abs(loss - prev_loss)
        if t % 10 == 0:
            print("Delta: " + str(delta_g))
        prev_loss = loss
        error = 0
        t = t + 1

    print(t)
    print(delta_g)
    return parameters

with open("training_exp1_first.csv") as f:
    rows = csv.reader(f, delimiter=",")

    data = []
    count0 = 0
    count1 = 0
    for row in rows:
        for i in range(len(row)):
            if row[i] == "*":
                row[i] = -1

        new_row = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
        data.append(new_row)
        # if new_row[3] == 1 and count1 is not 635:
        #     data.append(new_row)
        # elif new_row[3] == 0:
        #     data.append(new_row)
        # if new_row[3] == 0:
        #     count0 += 1
        # else:
        #     count1 = min(count1+1,635)


    count0 = 0
    count1 = 0
    for new_row in data:
        if new_row[3] == 0:
            count0 += 1
        else:
            count1 += 1

    print(count0)
    print(count1)
    lr = 10**-4
    delta = 10**-5

    parameters = ICLogisticRegression(data, lr, delta)
    print("Param " + str(parameters))
    # samples = [[5,5,5],[4,4,4],[3,3,3],[2,2,2],[1,1,1],[0,0,0]]
    # for item in samples:
    #     k = parameters[0] + item[0]*parameters[1] + item[1]*parameters[2] + item[2]*parameters[3]
    #     value = (1 / (1 + math.exp(-k)))
    #     if value >= 0.5:
    #         pred_label = 1
    #     else:
    #         pred_label = 0
        
    #     print(value)
    #     print(pred_label)

    pred0 = 0
    pred1 = 1
    num_error = 0
    # for item in data:
    #     k = parameters[0] + item[0]*parameters[1] + item[1]*parameters[2] + item[2]*parameters[3]
    #     value = (1 / (1 + math.exp(-k)))
    #     if value >= 0.5:
    #         pred1 += 1
    #         sum = 0
    #         for i in item[:-1]:
    #             sum += i
    #         sum = sum/3
    #         if (sum < 2):
    #             num_error += 1
    #     else:
    #         pred0 += 1
    #         sum = 0
    #         for i in item[:-1]:
    #             sum += i
    #         sum = sum/3
    #         if (sum >= 2):
    #             num_error += 1
        
    
    # print(pred0)
    # print(pred1)
    # print(num_error)

    with open("pred0.csv","w", newline="\n") as csv0:
        writer0 = csv.writer(csv0, delimiter=",")
        with open("pred1.csv","w", newline="\n") as csv1:
            writer1 = csv.writer(csv1, delimiter=",")
            for item in data:
                k = parameters[0] + item[0]*parameters[1] + item[1]*parameters[2] + item[2]*parameters[3]
                value = (1 / (1 + math.exp(-k)))
                if value >= 0.5:
                    writer1.writerow(item)
                else:
                    writer0.writerow(item)