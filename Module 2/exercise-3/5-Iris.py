import pandas as pd
import numpy as np

def compute_prior(train_data):
    y_sample = ["setosa","versicolor","virginica"]
    prior = np.zeros(shape=len(y_sample))
    for i in range( len(train_data) ):
        if train_data[i,4] == y_sample[0]:
            prior[0] += 1
        elif train_data[i,4] == y_sample[1]:
            prior[1] += 1
        else:
            prior[2] += 1
    return prior/len(train_data)

def compute_p(train_data):
    y_sample = ["setosa","versicolor","virginica"]
    list_name = []
    p = []
    for i in range(train_data.shape[1]-1):
        col_unique = np.unique(train_data[:,i])
        print(i," ",col_unique )
        list_name.append( col_unique )
        temp = []
        for y in y_sample:
            l = []
            for x in col_unique:
                count_x = np.count_nonzero((train_data[:,i]==x) & (train_data[:,4]==y))
                count_y = np.count_nonzero(train_data[:,4]==y)
                l.append(count_x/count_y)
            temp.extend( [l] )
        p.extend( [temp] )
    return p,list_name

def find_index(list_name,id,i):
    return np.where( list_name[i] == id)[0][0]

def prediction(X,p,listname,prior):
    y_sample = ["setosa", "versicolor", "virginica"]
    result = []
    for i in range(len(y_sample)):
        temp = 1
        for j in range(len(X)):
            pos = find_index(listname,X[j],j)
            temp = temp * p[j][i][pos]
        temp = temp*prior[i]
        result.append( temp )
    print("The result is: ", y_sample[ result.index( max(result) ) ])


df = pd.read_csv("iris.data.txt")
df.iloc[:,4] = df.iloc[:,4].apply(lambda x: x.replace("Iris-",""))
train_data = df.to_numpy()
prior = compute_prior( train_data )
p,list_name = compute_p(train_data)
print( p )
X = [6.3,3.3,6.0,2.5]
prediction(X,p,list_name,prior)
X = [5.0,2.0,3.5,1.0]
prediction(X,p,list_name,prior)
X = [4.9,3.1,1.5,0.1]
prediction(X,p,list_name,prior)


