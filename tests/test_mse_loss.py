import torch
import torch.nn as nn

loss = nn.MSELoss()
input = torch.randn(1, 2, 3, requires_grad=True)
target = torch.randn(1, 2, 3)
output = loss(input, target)
print(input)
print(target)
print(output)
print((input-target)**2)