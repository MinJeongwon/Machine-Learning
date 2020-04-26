import tensorflow as tf

# build graph
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2) # or node3 = node1 + node2

# Session() : not available in tf 2.0
print('node1 : ', node1, '\nnode2 : ', node2, '\nnode3 : ', node3)

# tensor
t1 = tf.constant([1, 2, 3])
t2 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
t3 = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
t4 = tf.constant(1)
print('t1 dimension : ', t1.ndim)
print('t1 shape : ', t1.shape)
print('t2 dimension : ', t2.ndim)
print('t2 shape : ', t2.shape)
print('t3 dimension : ', t3.ndim)
print('t3 shape : ', t3.shape)
print('t4 dimension :', t4.ndim)
print('t4 shape : ', t4.shape)



