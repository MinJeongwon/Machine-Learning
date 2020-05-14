import tensorflow as tf
import numpy as np

X = tf.constant([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]], dtype = tf.float32, name = 'input')

W1 = tf.constant([[5, -7],
                 [5, -7]], dtype = tf.float32, name = 'weight1')
b1 = tf.constant([-8, 3], dtype = tf.float32, name = 'bias1')

K = tf.sigmoid(tf.matmul(X, W1) + b1)
print(K)

W2 = tf.constant([[-11],
                 [-11]], dtype = tf.float32, name = 'weight2')
b2 = tf.constant([6], dtype = tf.float32, name = 'bias2')

Y = tf.sigmoid(tf.matmul(K, W2) + b2)
print(Y)

result = tf.cast(Y > 0.5, dtype = tf.int32)
print(result)

