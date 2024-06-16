import torch
import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)

class SoftmaxStable(nn.Module):
    def __init__(self):
        super(SoftmaxStable, self).__init__()

    def forward(self, x):
        c = torch.max(x)
        return torch.exp(x - c) / torch.sum(torch.exp(x - c), dim=-1, keepdim=True)

data = torch.Tensor([1,2,3])
softmax = Softmax()
output = softmax(data)
print("The result of softmax is: {}".format(output))

data = torch.Tensor([1,2,3])
softmax_stable = SoftmaxStable()
output = softmax_stable(data)
print("The result of softmax_stable is: {}".format(output))
