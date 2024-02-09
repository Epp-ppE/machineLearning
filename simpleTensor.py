import tensorflow as tf
print(tf.__version__) 

# Creating tensors

# Data Types Include: float32, int32, string and others.
# Shape: Represents the dimension of data.

rank1_tensor = tf.Variable(["Test"], dtype=tf.string)
rank2_tensor = tf.Variable([[1,2,3],[4,5,6]], dtype=tf.float16)

print(rank1_tensor)
print(rank2_tensor)

# To determine the rank of a tensor
# print(tf.rank(rank1_tensor))
# print(tf.rank(rank2_tensor))

# To determine the shape of a tensor
# print(rank1_tensor.shape)
# print(rank2_tensor.shape)

# Changing shape of tensors
# Shape of a tensor is the number of elements in each dimension.
# The length of the shape is the rank of the tensor.
tensor1 = tf.ones([1,2,3])
# print(tensor1)
tensor2 = tf.reshape(tensor1, [2,3,1])
# print(tensor2)
tensor3 = tf.reshape(tensor2, [3,-1])
# print(tensor3)

# Operations on tensors
# Addition 
tensor4 = tensor3 + 10
# print(tensor4)

#Evaluation of tensors
print(tensor4.numpy())
