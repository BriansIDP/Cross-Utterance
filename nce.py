import torch
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn import functional as F

class nce_loss(Module):
    def __init__(self, size_average = True):
        super(nce_loss, self).__init__()
        self.size_average = size_average

    def forward(self, input):
        logits, nce_target = input
        N, Kp1 = logits.size()  # num true x (num_noise+1)

        loss = F.binary_cross_entropy_with_logits(logits, nce_target, reduce = False)

        loss = torch.sum(loss)

        if self.size_average:
            loss /= N

        return loss
