import torch

# "需要:初始化h0,输入序列"
batch_size = 1
input_size = 4
hidden_size = 2
seq_len = 3

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

dataset = torch.randn(seq_len, batch_size, input_size)  # 构造输入序列
hidden = torch.zeros(batch_size, hidden_size)  # 构造全是0的隐层,即初始化h0

for idex, input in enumerate(dataset):
    print('=' * 20, idex, '=' * 20)
    print('Input size:', input.shape)
    hidden = cell(input, hidden)
    print('outputs size:', hidden.shape)
    print('hidden:', hidden)