import pandas as pd
import numpy as np

def correlation(X,Y):
    N = len(X)
    return ( (N*np.sum(X*Y) - np.sum(X)*np.sum(Y) )
            /((np.sqrt(N*np.sum(X**2)-(np.sum(X)**2)))*np.sqrt(N*np.sum(Y**2)-(np.sum(Y)**2))) )

df = pd.read_csv("advertising.csv")
x = df["TV"].to_numpy()
y = df["Radio"].to_numpy()
print( correlation(x,y) )

features = ["TV","Radio","Newspaper"]
for i in features:
    for j in features:
        corr = correlation(df[i].to_numpy(),df[j].to_numpy())
        print("{} and {}: {}".format(i,j,round(corr,2)))

print( df.corr() )