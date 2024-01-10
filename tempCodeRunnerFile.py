import numpy as np

def change(Matrix):
    Matrix[0][0] = 66
    return Matrix

# Create a NumPy array
original_matrix = np.array([[0, 2], [3, 4]])

# Call the function and modify the array
modified_matrix = change(original_matrix)

print("Original Matrix:")
print(original_matrix)
print("Modified Matrix:")
print(modified_matrix)
