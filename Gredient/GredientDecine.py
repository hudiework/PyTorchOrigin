import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        loss = (y_pred - y) ** 2
        cost += loss
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
        return grad / len(xs)


cost_list = []
epoch_list = []

print("predict (before trainning)", 4, forward(4))

for epoch in range(100):
    epoch_list.append(epoch)
    cost_val = cost(x_data, y_data)
    cost_list.append(cost_val)
    gradient_val = gradient(x_data, y_data)
    w -= 0.01 * gradient_val
    print("Epoch:", epoch, "w=", w, "loss = ", cost_val)

print("Prediction (after trainning)", 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()
