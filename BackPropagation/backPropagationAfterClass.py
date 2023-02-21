import  torch
import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.tensor([1.0], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)


def forward(x):
    return (x**2) * w1 + x * w2 +b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

cost_list = []
epoch_list = []

print("Predict (before):", 4 , forward(4).item())

for epoch in range(100):
    epoch_list.append(epoch)
    for x, y in zip(x_data,y_data):
        l= loss(x,y)
        l.backward()
        print("\t grad:",x,y,w1.grad.item(),w2.grad.item(),b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w1.grad.data.zero_()
        w2.data = w2.data - 0.01 * w2.grad.data
        w2.grad.data.zero_()
        b.data = b.data - 0.01 * b.grad.data
        b.grad.data.zero_()
    print('Progress:', epoch, l.item())
    cost_list.append(l.item())
print("Predict (after)", 4, forward(4).item())
plt.plot(epoch_list, cost_list)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()