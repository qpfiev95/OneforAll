import numpy as np

# 1D array - vector
arr_1d_1 = np.array([1.0, 2.3, 3.2])
arr_1d_2 = np.array([1.0, 2.3, 3.2])

# 2D array - matrix 2x3
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

#print(f"arr_1d info: {arr_1d.shape}, {arr_1d.size}, {arr_1d.dtype}")
print(f"arr_2d info: {arr_2d.shape}, {arr_2d.size}, {arr_2d.dtype}")

## Operations
# + - * \
plus = arr_1d_1 + arr_1d_2
print(plus[0])
print(np.mean(plus))
print(np.max(plus))

# max multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B)) 
