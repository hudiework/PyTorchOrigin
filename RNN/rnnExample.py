import torch
import torch.nn as nn
import torch.optim as optim

input_size = 4
hidden_size = 4
batch_size = 1

# prepare data
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # hello 输入
y_data = [3, 1, 2, 3, 2]  # ohlol 目标

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]  # 分别对应0，1，2，3即e,h,l,o 独热编码

x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)  # -1即seqLen
labels = torch.LongTensor(y_data).view(-1, 1)  # (seqLen,1)


# define model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnncell = nn.RNNCell(input_size=self.input_size,
                                  hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


model = Model(input_size, hidden_size, batch_size)

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# training cycle
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = model.init_hidden()  # h0
    print('predicted string:', end='')
    for input, label in zip(inputs, labels):
        hidden = model(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)  # hidden是4维的，分别表示e,h,l,o的概率值
        print(idx2char[idx.item()], end='')

    loss.backward()
    optimizer.step()
    print(',epoch [%d/15] loss = %.4lf' % (epoch + 1, loss.item()))