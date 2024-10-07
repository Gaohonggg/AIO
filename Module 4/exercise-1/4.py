import numpy as np
import matplotlib.pyplot as plt
import random

def get_column(data,index):
    return [ row[index] for row in data ]

def prepare_data(file_name):
    data = np.genfromtxt("advertising.csv",delimiter=',',skip_header=1).tolist()

    tv_data = get_column(data,0)
    radio_data = get_column(data,1)
    newspaper_data = get_column(data,2)
    sales_data = get_column(data,3)

    X = [[1,x1,x2,x3] for x1, x2, x3 in zip(tv_data,radio_data,newspaper_data)]
    y = sales_data

    return X,y

def initialize_params():
    w1 = random.gauss(mu=0, sigma=0.01)
    w2 = random.gauss(mu=0, sigma=0.01)
    w3 = random.gauss(mu=0, sigma=0.01)
    b = 0
    return [0, -0.01268850433497871, 0.004752496982185252, 0.0073796171538643845]

def predict(X_features, weights):
    result = np.array( [x*w for x, w in zip(X_features,weights)] ).sum()
    return result

def compute_loss(y_hat,y):
    return (y_hat-y)**2

def compute_gradient_w(X_features, y, y_hat):
    X_features = np.array( X_features.copy() )
    dl_dweights = 2*(y_hat - y)*X_features
    dl_dweights[3] /= 2
    dl_dweights.tolist()
    return dl_dweights

def update_weight(weights, dl_dweights, lr):
    weights -= lr*np.array(dl_dweights)
    return weights.tolist()

def implement(X_feature, y_output, epoch_max=50, lr=1e-5):
    losses = []
    weights = initialize_params()
    N = len(y_output)

    for epoch in range(epoch_max):
        print("epoch",epoch)
        for i in range(N):
            features_i = X_feature[i]
            y = y_output[i]

            y_hat = predict(features_i,weights)
            loss = compute_loss(y_hat,y)

            dl_dw = compute_gradient_w(features_i,y,y_hat)
            weights = update_weight(weights,dl_dw,lr)

            losses.append(loss)
    return weights,losses

X , y = prepare_data("advertising.csv")
W , L = implement(X , y )
print ( L [9999])





































