# coding: utf-8
import argparse
import sys, os
import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Level-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/AMI',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='model.pt',
                    help='location of the 1st level model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--reset', action='store_true',
                    help='reset at sentence boundaries')
parser.add_argument('--memorycell', action='store_true',
                    help='Use memory cell as input, otherwise use output cell')
parser.add_argument('--uttlookback', type=int, default=1,
                    help='Number of utterance embeddings to be incorporated')
parser.add_argument('--saveprefix', type=str, default='tensors/AMI',
                    help='Specify which data utterance embeddings saved')
args = parser.parse_args()

# Read in dictionary
print("Reading dictionary...")
dictionary = {}
with open(os.path.join(args.data, 'dictionary.txt')) as vocabin:
    lines = vocabin.readlines()
    for line in lines:
        ind, word = line.strip().split(' ')
        if word not in dictionary:
            dictionary[word] = ind
        else:
            print("Error! Repeated words in the dictionary!")

eosidx = int(dictionary['<eos>'])

device = torch.device("cuda" if args.cuda else "cpu")

# Read in trained 1st level model
print("Reading model...")
with open(args.model, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()
    if args.cuda:
        model.cuda()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_utt_embedding_groups(model):
    model.eval()
    model.set_mode('eval')
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for setname in ['train', 'valid', 'test']:
            with open(os.path.join(args.data, setname+'.txt')) as fin:
                lines = fin.readlines()
                utt_embeddings = []
                totalfile = []
                prevemb = [0 for _ in range(args.uttlookback)]
                embind = []
                for ind, line in enumerate(lines):
                    currentline = []
                    for word in line.strip().split(' '):
                        currentline.append(int(dictionary[word]))
                    currentline.append(eosidx)
                    input = torch.LongTensor(currentline)
                    input = input.view(1, -1).t().to(device)
                    rnnout, hidden = model(input, hidden, outputflag=1)
                    if ind > args.uttlookback - 1:
                        utt_embeddings.append(torch.cat(prevemb, 2))
                        totalfile += currentline
                        embind += [ind - args.uttlookback for _ in range(len(currentline))]
                    if args.memorycell:
                        for i in range(args.uttlookback-1):
                            prevemb[i] = prevemb[i+1]
                        prevemb[args.uttlookback-1] = hidden[1]
                    else:
                        for i in range(args.uttlookback-1):
                            prevemb[i] = prevemb[i+1]
                        prevemb[args.uttlookback-1] = rnnout[-1,:,:]
                    if args.reset:
                        hidden = model.init_hidden(1)
                    else:
                        repackage_hidden(hidden)
                    if ind % 1000 == 0 and ind != 0:
                        print('{}/{} completed'.format(ind, len(lines)))
            torch.save(torch.cat(utt_embeddings, 0), args.saveprefix+setname+'_utt_embed.pt')
            torch.save(torch.LongTensor(totalfile), args.saveprefix+setname+'_fullind.pt')
            torch.save(torch.LongTensor(embind), args.saveprefix+setname+'_embind.pt')

print('getting utterances')
get_utt_embedding_groups(model)
