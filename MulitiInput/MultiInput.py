import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

xy = np.loadtxt("./diabetes.csv.gz", delimiter=",", dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.sigmod = nn.Sigmoid()


    def forward(self, x):
        x = self.sigmod(self.linear1(x))
        x = self.sigmod(self.linear2(x))
        x = self.sigmod(self.linear3(x))
        return x

model = Model()
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(),lr=0.1)
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('----------Number1------------')
print(model.linear1.weight.data)
print(model.linear1.bias.data)
print('----------Number2------------')
print(model.linear2.weight.data)
print(model.linear2.bias.data)
print('----------Number3------------')
print(model.linear3.weight.data)
print(model.linear3.bias.data)