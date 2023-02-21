import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(x, y):
    y_pred = forward(x)
    return 2 * x * (y_pred - y)


lost_list = []
epoch_list = []

print("predict (before trainning)", 4, forward(4))

for epoch in range(100):
    epoch_list.append(epoch)
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)
        grad = gradient(x, y)
        w -= 0.01 * grad
    print("Epoch:", epoch, "w=", w, "loss = ", loss_val)
    lost_list.append(loss_val)

print("Prediction (after trainning)", 4, forward(4))
plt.plot(epoch_list, lost_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
