import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class PredictorWN_deep(nn.Module):
    def __init__(self, inc=4096, num_emb=512, num_class=64, temp=0.05):
        super(PredictorWN_deep, self).__init__()
        self.fc1 = nn.Linear(inc, num_emb)
        self.bn = nn.BatchNorm1d(num_emb)
        self.relu = nn.ReLU(inplace=True)
        self.proxy = torch.nn.Parameter(torch.randn(num_class, num_emb).cuda())
        nn.init.kaiming_normal_(self.proxy, mode='fan_out')

        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=1.0, getemb=False, normemb=True):
        featwonorm = self.fc1(x)
        featwonorm = self.bn(featwonorm)
        featwonorm = self.relu(featwonorm)
        emb = F.normalize(featwonorm)
        proxy = l2_norm(self.proxy)
        cos = F.linear(emb, proxy)
        x_out = cos / self.temp
        if getemb:
            remb = emb
            return x_out, remb
        else:
            return x_out