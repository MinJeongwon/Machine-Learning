import tensorflow as tf 
tf.random.set_seed(0)

# data and label
x1 = [ 73., 93., 89., 96., 73. ]
x2 = [ 80., 88., 91., 98., 66. ]
x3 = [ 75., 93., 90., 100., 70. ]
Y = [152., 185., 180., 196., 142.]

# weights
w1 = tf.Variable(tf.random.normal([1]))
w2 = tf.Variable(tf.random.normal([1]))
w3 = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))

# learning rate
lr = 0.000001

# model
hypothesis = w1*x1 + w2*x2 + w3*x3 + b 
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# learn
for i in range(1000+1):
	with tf.GradientTape() as tape:
		hypothesis = w1*x1 + w2*x2 + w3*x3 + b 
		cost = tf.reduce_mean(tf.square(hypothesis - Y))
	w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

	# update w1, w2, w3, b  
	w1.assign_sub(lr * w1_grad)
	w2.assign_sub(lr * w2_grad)
	w3.assign_sub(lr * w3_grad)
	b.assign_sub(lr * b_grad)

	if i % 50 == 0:
		print("{:5} | {:12.4f}".format(i, cost.numpy()))