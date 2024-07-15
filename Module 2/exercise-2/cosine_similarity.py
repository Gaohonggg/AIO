import numpy as np
from muilti_lenght_matrix_vector import compute_vector_length,compute_dot_product

def compute_cosine(v1,v2):
    if len(v1) == len(v2):
        result = (compute_dot_product(v1,v2)
                  /(compute_vector_length(v1)*compute_vector_length(v2)))
        return result
    return "Fail"

print( compute_cosine(np.array([1,2,3,4]),np.array([1,0,3,0])) )