import torch

a = torch.LongTensor([1,2,3])
b = torch.Tensor([1,2,3,4]).view(2, -1)

print(a.size())

print(b.size())

print('The end!')