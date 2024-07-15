import numpy as np

def compute_eigenvalues_eigenvectors(matrix):
    return np.linalg.eig(matrix)

matrix = np.array( [[0.9,0.2],[0.1,0.8]] )
eigv,eigvec = compute_eigenvalues_eigenvectors(matrix)
print(eigv)
print(eigvec)