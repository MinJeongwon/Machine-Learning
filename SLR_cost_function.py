# cost function

import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

def cost_func(W, X, Y):
    c = 0
    for i in range(len(X)):
        c += (W*X[i] - Y[i]) ** 2
    return c / len(X)

# np.linspace : returns an array with num evenly spaced samples, calculated over the interval [start, stop]
# >>>>> np.linspace(2.0, 3.0, num=5)
# array([2.  , 2.25, 2.5 , 2.75, 3.  ])

feed_W_list =[]
cost_list =[]

for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = cost_func(feed_W, X, Y)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

    feed_W_list.append(feed_W)
    cost_list.append(curr_cost)

plt.plot(feed_W_list, cost_list, label = 'cost')
plt.title('cost function')
plt.legend()
plt.show()
