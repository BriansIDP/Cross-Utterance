import sys, os
from torch.utils.data import Dataset, DataLoader
import torch

class Dictionary(object):
    def __init__(self, dictfile):
        self.word2idx = {}
        self.idx2word = []
        self.unigram = []
        self.build_dict(dictfile)

    def build_dict(self, dictfile):
        with open(dictfile, 'r', encoding="utf8") as f:
            for line in f:
                index, word = line.strip().split(' ')
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.unigram.append(0)
        self.unigram[self.word2idx[word]] += 1
        return self.word2idx[word]

    def get_eos(self):
        return self.word2idx['<eos>']

    def normalize_counts(self):
        self.unigram /= np.sum(self.unigram)
        self.unigram = self.unigram.tolist()

    def __len__(self):
        return len(self.idx2word)

class LMdata(Dataset):
    def __init__(self, data_file, dictionary):
        '''Load data_file'''
        self.data_file = data_file
        self.data = []
        with open(self.data_file, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                self.data += words
        self.dictionary = dictionary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data[idx] in self.dictionary.word2idx:
            return self.dictionary.word2idx[self.data[idx]]
        else:
            return self.dictionary.word2idx['OOV']

def collate_fn(batch):
    return torch.LongTensor(batch)

def create(datapath, dictfile, batchSize=1, shuffle=False, workers=0):
    loaders = []
    dictionary = Dictionary(dictfile)
    for split in ['train', 'valid', 'test']:
        data_file = os.path.join(datapath, '%s.txt' %split)
        dataset = LMdata(data_file, dictionary)
        loaders.append(DataLoader(dataset=dataset, batch_size=batchSize,
                                  shuffle=shuffle, collate_fn=collate_fn,
                                  num_workers=workers))
    return loaders[0], loaders[1], loaders[2]

if __name__ == "__main__":
    datapath = sys.argv[1]
    dictfile = sys.argv[2]
    traindata, valdata, testdata = create(datapath, dictfile, batchSize=1000000, workers=0)
    for i_batch, sample_batched in enumerate(traindata):
        print(i_batch, sample_batched.size())
