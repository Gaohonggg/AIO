import pandas as pd
import numpy as np

df = pd.read_csv("iris.data.txt",header=None)
df.iloc[:,4] = df.iloc[:,4].apply(lambda x: x.replace("Iris-",""))
train_data = df.to_numpy()

def compute_p_A(train_data):
    set = ["setosa","versicolor","virginica"]
    p = []
    for i in set:
        p.append( np.count_nonzero( train_data[:,4]==i ) )
    return np.array(p)/len(train_data)

def expect(train_data):
    result = 0
    for i in train_data:
        result += i
    return result / len(train_data)

def var(train_data,expect):
    result = 0
    for i in train_data:
        result += (i - expect)**2
    return result / len(train_data)

def compute_e_and_v(train_data):
    set = ["setosa","versicolor","virginica"]
    p = []
    for i in set:
        l = [[],[]]
        for j in range(4):
            temp = train_data[ train_data[:,4]==i,j ]
            e = expect(temp)
            v = var(temp,e)
            l[0].append( e )
            l[1].append( v )
        p.extend( [l] )
    return np.array(p)

def gauss(x,i,j,e_v):
    k = 1/(np.sqrt(e_v[i,1,j])*np.sqrt(2*3.14))
    f = (x-e_v[i,0,j])**2 / (2*e_v[i,1,j])
    return k * np.exp(-f)

def compute(X,p_A,e_v):
    set = ["setosa", "versicolor", "virginica"]
    result = []
    for i in range(len(set)):
        temp = 1
        for j in range(len(X)):
            temp = temp * gauss(X[j],i,j,e_v)
        result.append( temp*p_A[i] )
    return set[ result.index( max(result) ) ]

p_A = compute_p_A(train_data)
print( p_A )
e_v = compute_e_and_v(train_data)
print( e_v )

X = [6.3,3.3,6.0,2.5]
print( compute(X,p_A,e_v) )
X = [5.0,2.0,3.5,1.0]
print( compute(X,p_A,e_v) )
X = [4.9,3.1,1.5,0.1]
print( compute(X,p_A,e_v) )