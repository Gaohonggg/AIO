import numpy as np
def compute_vector_length(vector):
    result = np.sqrt( np.sum( vector**2 ) )
    if result == np.linalg.norm(vector):
        return result
    return "Fail"

def compute_dot_product(vector1,vector2):
    if len(vector1) == len(vector2):
        result = np.sum( vector1*vector2 )
        if result == np.dot(vector1,vector2):
            return result
    return "Fail"

def matrix_multi_vector(matrix,vector):
    if matrix.shape[1] == len(vector):
        result = np.sum( matrix*vector,axis=1 )
        if  np.array_equal(result,np.dot(matrix,vector)):
            return result
    return "Fail"

def matrix_multi_matrix(matrix1,matrix2):
    if matrix1.shape[1] == matrix2.shape[0]:
        result = np.zeros((matrix1.shape[0],matrix2.shape[1]))
        for i in range( matrix1.shape[0] ):
            for j in range( matrix2.shape[1] ):
                result[i,j] = np.sum( matrix1[i,:]*matrix2[:,j] )
        if np.array_equal(result,np.dot(matrix1,matrix2)):
            return result
    return "Fail"

def inverse_matrix(matrix):
    if np.linalg.det(matrix)==0:
        return "Error"
    return np.linalg.inv(matrix)

vec = np.array([3,4,5])
print("The len of the vec is {}".format(compute_vector_length(vec)))
vec2 = np.array([1,2,3])
print("The dot of the vec is {}".format(compute_dot_product(vec,vec2)))
matrix = np.array([[1,1,1],[2,2,2],[3,3,3]])
print(matrix_multi_vector(matrix,vec))
matrix1 = np.array([[1,1],[2,2],[3,3]])
print(matrix_multi_matrix(matrix,matrix1))
maxtrix3 = [[-2,6],[8,-4]]
print(inverse_matrix(maxtrix3))
