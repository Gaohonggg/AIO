import numpy as np

def create_polynomial_features(X, degree=2):
    X_new = X
    for d in range(2,degree+1):
        X_new = np.c_[X_new, np.power(X,d)]
    return X_new

X = np.array([[1] , [2] , [3]])
print( create_polynomial_features(X) )