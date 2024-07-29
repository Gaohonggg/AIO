import numpy as np

def compute_mean(X):
    return np.sum(X)/len(X)

def compute_median(X):
    X = np.sort(X)
    if len(X)%2!=0:
        return X[(len(X)+1)/2]
    return 0.5 * (X[len(X)//2]+X[(len(X)//2)-1])

def compute_std(X):
    mean = compute_mean(X)
    var = 1/len(X) * np.sum( (X-mean)**2 )
    return np.sqrt( var )

def compute_correlation_cofficient(X,Y):
    N = len(X)
    return ( (N*np.sum(X*Y) - np.sum(X)*np.sum(Y) )
            /((np.sqrt(N*np.sum(X**2)-(np.sum(X)**2)))*np.sqrt(N*np.sum(Y**2)-(np.sum(Y)**2))) )

X = [2,0,2,2,7,4,-2,5,-1,-1]
print( compute_mean(X) )

X = [1,5,4,4,9,13]
print( compute_median(X) )

X = [171,176,155,167,169,182]
print( compute_std(X) )

X = [-2,-5,-11,6,4,15,9]
Y = [4,25,121,36,16,225,81]
print( compute_correlation_cofficient(np.array(X),np.array(Y)) )