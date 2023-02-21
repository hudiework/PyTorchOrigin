import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0]])

class LinearModel(nn.Module):
    def  __init__(self):
        super(LinearModel,self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
criterion = nn.MSELoss(size_average=False)
# 小批量损失求和

optimizer = optim.SGD(model.parameters(),lr=0.01)
# optimizer = optim.Adagrad(model.parameters(),lr=0.01)
# optimizer = optim.Adam(model.parameters(),lr=0.01)
# optimizer = optim.Adamax(model.parameters(),lr=0.01)
# optimizer = optim.ASGD(model.parameters(),lr=0.01)
# optimizer = optim.LBFGS(model.parameters(),lr=0.01)
# optimizer = optim.RMSprop(model.parameters(),lr=0.01)
# optimizer = optim.Rprop(model.parameters(),lr=0.01)

epoch_list=[]
loss_list=[]

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss)
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w= ',model.linear.weight.item())
print('b= ',model.linear.bias.item())


x_test = torch.tensor([4.0])
y_test = model(x_test)
print('y_pred= ',y_test.item())

plt.plot(epoch_list, loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()