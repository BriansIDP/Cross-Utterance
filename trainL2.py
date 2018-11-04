# coding: utf-8
import time
import argparse
import sys, os
import torch
import torch.nn as nn

import data
from L2model import L2RNNModel

arglist = []
parser = argparse.ArgumentParser(description='PyTorch Level-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/AMI',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--naux', type=int, default=128,
                    help='auxiliary context info feature dimension (after compressor)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--save', type=str, default='model.pt',
                    help='location of the model to be saved')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--reset', action='store_true',
                    help='reset at sentence boundaries')
parser.add_argument('--batchsize', type=int, default=32,
                    help='Batch size used for training')
parser.add_argument('--bptt', type=int, default=35,
                    help='bptt steps used for training')
parser.add_argument('--saveprefix', type=str, default='tensors/AMI',
                    help='Specify which data utterance embeddings saved')
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")

arglist.append(('Data', args.data))
arglist.append(('Model', args.model))
arglist.append(('Embedding Size', args.emsize))
arglist.append(('Auxiliary Input Size', args.naux))
arglist.append(('Hidden Layer Size', args.nhid))
arglist.append(('Layer Number', args.nlayers))
arglist.append(('Learning Rate', args.lr))
arglist.append(('Update Clip', args.clip))
arglist.append(('Max Epochs', args.epochs))
arglist.append(('BatchSize', args.batchsize))
arglist.append(('Sequence Length', args.bptt))
arglist.append(('Dropout', args.dropout))

def read_in_dict():
    # Read in dictionary
    print("Reading dictionary...")
    dictionary = {}
    with open(os.path.join(args.data, 'dictionary.txt')) as vocabin:
        lines = vocabin.readlines()
        ntoken = len(lines)
        # for line in lines:
        #     ind, word = line.strip().split(' ')
        #     if word not in dictionary:
        #         dictionary[word] = ind
        #     else:
        #         print("Error! Repeated words in the dictionary!")
    return ntoken

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, ind, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    embind = ind[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, embind, target, seq_len

def load_utt_embeddings():
    return torch.load(args.saveprefix+'_utt_embed.pt'), torch.load(args.saveprefix+'_fullind.pt'), torch.load(args.saveprefix+'_embind.pt')

def batchify(data, embind, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    embind = embind.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    embind = embind.view(bsz, -1).t().contiguous()
    return data.to(device), embind.to(device)

def fill_uttemb_batch(utt_embeddings, embind, bsz, bptt):
    embind = embind.view(-1)
    batched_utt_embeddings = torch.index_select(utt_embeddings, 0, embind)
    return batched_utt_embeddings.view(bptt, bsz, -1)

def train(traindata, utt_embeddings, embind_batched, model):
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, traindata.size(0) - 1, args.bptt)):
        data, ind, targets, seq_len = get_batch(traindata, embind_batched, i)
        auxinput = fill_uttemb_batch(utt_embeddings, ind, args.batchsize, seq_len)
        print('At position {}'.format(i))
        

# ---------------------
# Main code starts here
# ---------------------
utt_embeddings, totalfile, embind = load_utt_embeddings()
data, embind_batched = batchify(totalfile, embind, args.batchsize)

# Model and optimizer instantiation
ntokens = read_in_dict()
model = L2RNNModel(args.model, ntokens, args.emsize, utt_embeddings.size(2), args.naux, args.nhid, args.nlayers, args.dropout, reset=args.reset).to(device)
criterion = nn.CrossEntropyLoss()

# Start training
print('Training Start!')
for pairs in arglist:
    print(pairs[0] + ': ', pairs[1])
# Loop over epochs.
lr = args.lr
best_val_loss = None
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(data, utt_embeddings.view(-1, utt_embeddings.size(2)), embind_batched, model)
        print('time elapsed is {:5.2f}s'.format((time.time() - epoch_start_time)))
        # val_loss = evaluate(val_data)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #         'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                    val_loss, math.exp(val_loss)))
        # print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        # if not best_val_loss or val_loss < best_val_loss:
        #     with open(args.save, 'wb') as f:
        #         torch.save(model, f)
        #     best_val_loss = val_loss
        # else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     lr /= 2.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
