from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

data = np.array([
    [2.0, 3.0, 1.5],
    [3.0, 3.5, 2.0],
    [3.5, 3.0, 2.5],
    [8.0, 8.0, 7.5],
    [8.5, 8.5, 8.0],
    [9.0, 8.0, 8.5],
    [1.0, 2.0, 1.0],
    [1.5, 2.5, 1.5],
])

df = pd.DataFrame(data,columns=["Feature1","Feature2","Feature3"])

def distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def initialize_centroids(data,k):
    return data[np.random.choice(data.shape[0],k,replace=False)]

def assign_clusters(data,centroids):
    dis = np.array( [[distance(x,centroid) for centroid in centroids] for x in data] )
    return np.argmin(dis,axis=1)

def update_centroids(data,clusters,k):
    return np.array( [data[clusters==i].mean(axis=0) for i in range(k)] )

k = 3
centroids = initialize_centroids(data,k)
for i in range(100):
    clusters = assign_clusters(data,centroids)
    new_centroids = update_centroids(data,clusters,k)
    if np.all( centroids == new_centroids ):
        break
    centroids = new_centroids

print( centroids )
print( clusters )







































