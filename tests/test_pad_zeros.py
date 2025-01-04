import torch
import torch.nn.functional as F

tensor = torch.randn((2,3,4))
desired_len = 6
padded_tensor = F.pad(tensor, (0, desired_len - tensor.shape[-1]))

print(tensor.shape)
print(padded_tensor.shape)

print(tensor)
print(padded_tensor)