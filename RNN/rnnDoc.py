import torch

batch_size = 1
input_size = 4
hidden_size = 2
seq_len = 3
num_layers = 2

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)

print('output size:', out.shape)
print('out:', out)
print('hidden size:', hidden.shape)
print('hidden:', hidden)