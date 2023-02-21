import torch
import torch.nn as nn
import torch.optim as optim

num_class = 4  # 4个类别，
input_size = 4  # 输入维度
hidden_size = 8  # 隐层输出维度，有8个隐层
embedding_size = 10  # 嵌入到10维空间
num_layers = 2  # 2层的RNN
batch_size = 1
seq_len = 5  # 序列长度5

# prepare data
idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]]  # (batch, seq_len) list
y_data = [3, 1, 2, 3, 2]  # ohlol

inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)


# define model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)


model = Model()

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# training cycle
for epoch in range(15):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('outputs:', outputs)
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()  # reshape to numpy
    print('idx', idx)
    print('Pridected:', ''.join([idx2char[x] for x in idx]), end='')  # end是不自动换行，''.join是连接字符串数组
    print(',Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))