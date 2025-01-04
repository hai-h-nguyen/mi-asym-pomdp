import torch
import torch.nn as nn


input1 = torch.randn(50, 100, 128)
input2 = torch.randn(50, 100, 128)
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
output = cos(input1, -input1)
print(output)