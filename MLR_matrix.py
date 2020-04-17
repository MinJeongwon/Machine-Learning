import tensorflow as tf 
import numpy as np 

tf.random.set_seed(0)

# data as matrix
data = np.array([
		[73., 80., 75., 152.],
		[93., 88., 93., 185.],
		[89., 91., 90., 180.],
		[96., 98., 100., 196.],
		[73., 66., 70., 142.]
	], dtype = np.float32)

# slice data using numpy
# [row, column]
X = data[:, :-1] 
y = data[:, [-1]]

W = tf.Variable(tf.random.normal([3, 1])) # 3row 1column
										  # X : [5, 3]
										  # y : [1, 1]
b = tf.Variable(tf.random.normal([1])) 

# type
print('type of W : ', type(W))
print('shape of W : ', W.shape)
print('type of b : ', type(b))
print('shape of b : ', b.shape)

# learning rate
lr = 0.000001

# hypothesis(prediction function)
def predict(X):
	return tf.matmul(X, W) + b

n_epochs = 2000
for i in range(n_epochs + 1):
	with tf.GradientTape() as tape:
		cost = tf.reduce_mean(tf.square(predict(X) - y))

	# get gradients
	W_grad, b_grad = tape.gradient(cost, [W, b])

	# update W, b
	W.assign_sub(lr * W_grad)
	b.assign_sub(lr * b_grad)

	if i % 100 == 0:
		print("{:5} | {:10.4f}".format(i, cost.numpy()))