import numpy as np
import matplotlib.pyplot as plt
import random

def get_column(data,index):
    return [ row[index] for row in data ]

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset,delimiter=',',skip_header=1).tolist()

    tv_data = get_column(data,0)
    radio_data = get_column(data,1)
    newspaper_data = get_column(data,2)
    sales_data = get_column(data,3)

    X = [tv_data,radio_data,newspaper_data]
    y = sales_data
    return X,y

X,y = prepare_data("advertising.csv")
list = [ sum(X[0][:5]),sum(X[1][:5]),sum(X[2][:5]),sum(y[:5]) ]
print(list)

def initialize_params():
    w1 = random.gauss(mu=0, sigma=0.01)
    w2 = random.gauss(mu=0, sigma=0.01)
    w3 = random.gauss(mu=0, sigma=0.01)
    b = 0
    w1, w2, w3, b = (0.016992259082509283, 0.0070783670518262355, -0.002307860847821344, 0)
    return w1, w2, w3, b

def predict(x1,x2,x3,w1,w2,w3,b):
    result = x1*w1 + x2*w2 + x3*w3 + b
    return result

def compute_loss(y_hat,y):
    loss = (y_hat-y)**2
    return loss

def compute_gradient_wi(xi,y,y_hat):
    dl_dwi = 2*(y_hat-y)*xi
    return dl_dwi

def compute_gradient_b(y,y_hat):
    dl_db = 2*(y_hat-y)
    return dl_db

def update_weight_wi(wi, dl_dwi, lr):
    wi = wi - lr*dl_dwi
    return wi

def update_weight_b(b,dl_db,lr):
    b = b - lr*dl_db
    return b

def implement_linear_regression(X_data,y_data,epoch_max=1000,lr=1e-5):
    losses = []
    w1, w2, w3, b = initialize_params()

    for epoch in range(epoch_max):
        loss_total = 0
        dw1_total = 0
        dw2_total = 0
        dw3_total = 0
        db_total = 0
        for i in range( len(y_data) ):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]
            y = y_data[i]

            y_hat = predict(x1, x2, x3, w1, w2, w3, b)
            loss = compute_loss(y,y_hat)
            loss_total += loss

            dw1_total += compute_gradient_wi(x1,y,y_hat)
            dw2_total += compute_gradient_wi(x2,y,y_hat)
            dw3_total += compute_gradient_wi(x3,y,y_hat)
            db_total += compute_gradient_b(y,y_hat)

        w1 = update_weight_wi(w1,dw1_total/len(y_data),lr)
        w2 = update_weight_wi(w2, dw2_total / len(y_data), lr)
        w3 = update_weight_wi(w3, dw3_total / len(y_data), lr)
        b = update_weight_wi(b, db_total / len(y_data), lr)
        losses.append(loss_total/len(y_data))
    return (w1,w2,w3,b,losses)

(w1,w2,w3,b,losses) = implement_linear_regression(X,y)
print( w1,w2,w3 )

plt.plot(losses)
plt.xlabel("#epoch")
plt.ylabel("MSE Loss")
plt.show()




























