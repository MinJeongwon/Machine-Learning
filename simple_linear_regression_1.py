import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# given data
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# initialize(randomly assign W and b)
W = tf.Variable(6.8)
b = tf.Variable(1.4)

learning_rate = 0.01

for i in range(100):
	with tf.GradientTape() as tape:
		hypothesis = W * x_data + b
		cost = tf.reduce_mean(tf.square(hypothesis - y_data))

	W_grad, b_grad = tape.gradient(cost, [W, b])

	# A.assign_sub(B)
	# A = A - B
	# Gradient descent
	W.assign_sub(learning_rate * W_grad)
	b.assign_sub(learning_rate * b_grad)

	if i % 10 == 0:
		print("{:5}|{:10.4}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

		plt.scatter(x_data, y_data, label = 'data')

		x = np.arange(0, 7, 1)
		y = [(W * num + b) for num in x]

		plt.xlim([0, 6])
		plt.ylim([0, 6])		

		plt.plot(x, y, label = 'W = '+ str(W.numpy()) + ',  b = ' + str(b.numpy()))
		plt.title('i = ' + str(i))
		plt.legend()
		plt.show()

# predict
print('predict')
print('x_data = 5 | y_data = ', W*5 + b)
print('x_data = 2.5 | y_data = ',W*2.5 + b)