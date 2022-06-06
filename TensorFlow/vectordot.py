import tensorflow as tf
import numpy as np

# Three very very simple ways of calculating the inner product
# using TensorFlow, Numpy, and from scratch.

# The two sample vectors for finding the inner product.
a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]

# Using Tensorflow's tensordot():
def vector_dot_tf(a, b):
    return tf.tensordot(a, b, 1)

# Using numpy's tensordot():
def vector_dot_np(a, b):
    return np.tensordot(a, b, 1)

# From scratch:
def vector_dot_scratch(a,b):
  if(np.shape(a) != np.shape(b)):
    print("Cannot do inner product as vectors are of distinct size.")
  else:
    c = 0
    for i in range(np.array(a).size):
      c += a[i-1]*b[i-1]
  return c
  
print(vector_dot_tf(a,b))
print(vector_dot_np(a,b))
print(vector_dot_scratch(a,b))