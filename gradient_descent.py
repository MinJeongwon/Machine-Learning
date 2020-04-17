import tensorflow as tf 
import numpy as np 

tf.random.set_seed(0)

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

W = tf.Variable(tf.random.normal([1], -100., 100.))

for step in range(300):
	hypothesis = W * x_data
	cost = tf.reduce_mean(tf.square(hypothesis - y_data))

	lr = 0.01
	gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - y_data, x_data))
	descent = W - tf.multiply(lr, gradient)

	# assign() : assigning new value into 'Variable' tensor
	W.assign(descent) 

	if step % 10 == 0:
		print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))


 