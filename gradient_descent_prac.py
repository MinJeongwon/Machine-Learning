import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

x_train = [1, 2, 3, 4, 5]
y_train = [2, 4, 6, 8, 10]

# convert data
# int -> float
# list -> array
x_train = np.array(x_train, dtype = np.float32)
y_train = np.array(y_train, dtype = np.float32)

# initialize W and b
W = tf.Variable(2.2)
b = tf.Variable(1.2)

# hypothesis
hypothesis = W * x_train + b

# cost function
cost_fn = tf.reduce_mean(tf.square(hypothesis - y_train))

# list for W and cost
W_list = []
cost_list = []

# learning rate
lr = 0.001

# gradient descent
for i in range(1001):
    with tf.GradientTape() as tape: 
        hypothesis = W * x_train + b
        curr_cost = tf.reduce_mean(tf.square(hypothesis - y_train))
    W_grad, b_grad = tape.gradient(curr_cost, [W, b])
    # descent
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    # train = optimizer.minimize(cost_fn)
   
    W.assign_sub(lr * W_grad)
    b.assign_sub(lr * b_grad)
    
    W_list.append(W.numpy())
    cost_list.append(curr_cost.numpy())

    if i % 100 == 0:
        print(i, W.numpy(), b.numpy(), curr_cost.numpy())

plt.subplot(211)
plt.scatter(x_train, y_train, color = 'blue', label = 'data')
plt.plot(x_train, hypothesis, color = 'red', label = 'fitted line')
plt.xlim(0, 6)
plt.ylim(0, 12)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(212)
plt.plot(W_list, cost_list, label = 'cost')
plt.xlabel('W')
plt.ylabel('cost')
plt.legend()
plt.show()
