import math

def ICLogisticRegression(data, lr, delta, parameters):
    t = 0 # Time step
    delta_g = float('inf') # Computed as abs(loss(t) - loss(t-1))
    error = 0  # Computed as abs(predicted_label - true_label)
    prev_loss = 0 # Loss(t-1)

    while delta_g > delta: # The training is performed until delta_g is less or equal than delta
        g = [0 for _ in range(len(parameters))] # Initializing g params

        for item in data: # for each element in the training set
            k = parameters[0] + item[0]*parameters[1] + item[1]*parameters[2] + item[2]*parameters[3] # compute k
            g[0] = g[0] + (1 / (1 + math.exp(-k))) - item[3] # update g0 

            for i in range(1,len(parameters)):
                g[i] = g[i] + (1 / (1 + math.exp(-k)) - item[3]) * item[i] #updates g1, g2, g3

        parameters[0] = parameters[0] - lr * g[0] # Updating beta_0 value
        for i in range(1, len(parameters)):
            parameters[i] = max(parameters[i] - lr * g[i], 0) # If a beta[i] value is less than 0, than beta[i] = 0, else beta[i] = beta[i] - lr * g[i]

        # Predicting the labels of the samples
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

    return parameters # return the beta values
