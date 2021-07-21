import math

def ICLogisticRegression(data, lr, delta, parameters):
    t = 0 # time step
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
        delta_g = abs(loss - prev_loss)
        prev_loss = loss
        error = 0
        t = t + 1

    return parameters