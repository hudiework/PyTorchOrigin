import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# prepare dataset, y=2x+3

x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 7.0, 9.0]

# 生成矩阵坐标
W, B = np.arange(0.0, 4.1, 0.1).round(1), np.arange(0.0, 4.1, 0.1).round(1)
w, b = np.meshgrid(W, B)


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

mse = l_sum / len(x_data)

# 绘图

fig = plt.figure()

ax = Axes3D(fig)
surf = ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap='rainbow')

# 设置下标

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')

# 设置颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
