import torch.nn as nn
from torch import cat, zeros, matmul, eye, set_printoptions
from torch.autograd import Variable

class SelfAttenModel(nn.Module):
    '''Implementation of self-attentive layer'''
    def __init__(self, ninp, ninterm, nweights):
        super(SelfAttenModel, self).__init__()
        self.layer1 = nn.Linear(ninp, ninterm, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.layer2 = nn.Linear(ninterm, nweights, bias=False)
        self.ninp = ninp
        self.nweights = nweights
        self.ninterm = ninterm
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer2.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, uttemb, scale=1, device='cuda', writeout=False):
        if uttemb.size(2) % self.ninp != 0:
            print('Splitting of input embedding is invalid!')
            raise
        totaloutput = zeros(uttemb.size(0),uttemb.size(1),self.ninp*self.nweights).to(device)
        penalty = zeros(uttemb.size(0)).to(device)
        for i in range(uttemb.size(0)):
            procunit = uttemb[i].view(uttemb.size(1), -1, self.ninp)
            intermediate = self.layer1(procunit)
            intermediate = self.tanh(intermediate)
            annotmatrix = self.layer2(intermediate)
            annotmatrix = self.softmax(annotmatrix)
            output = matmul(annotmatrix.transpose(1,2), procunit)
            totaloutput[i] = output.view(output.size(0), output.size(1)*output.size(2))
            ATA = matmul(annotmatrix.transpose(1,2), annotmatrix)
            I = eye(self.nweights).to(device)
            penalty[i] = scale * ((ATA.to(device) - I.expand_as(ATA)) ** 2).sum()
        return totaloutput, penalty.sum() 
