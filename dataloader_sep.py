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
    def __init__(self, data_file, dictionary, individual_utt = False):
        '''Load data_file'''
        self.data_file = data_file
        self.data = []
        self.uttlist = []
        self.targetlist = []
        self.lenlist = []
        self.maxlen = 0
        self.dictionary = dictionary
        self.individual_utt = individual_utt
        with open(self.data_file, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                self.data += words
                if self.maxlen < len(words):
                    self.maxlen = len(words)
                if individual_utt:
                    self.uttlist.append(['<eos>']+words[0:-1])
                    self.targetlist.append(words)
                    self.lenlist.append(len(words))

    def __len__(self):
        if self.individual_utt:
            return len(self.uttlist)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.individual_utt:
            utt_ids = [0 for i in range(self.maxlen)]
            target_ids = [0 for i in range(self.maxlen)]
            for index, word in enumerate(self.uttlist[idx]):
                utt_ids[index] = self.dictionary.word2idx[word]
                if index > 0:
                    target_ids[index-1] = self.dictionary.word2idx[word]
            target_ids[index] = self.dictionary.word2idx['<eos>']
            return (utt_ids, target_ids, self.lenlist[idx])
        else:
            if self.data[idx] in self.dictionary.word2idx:
                return self.dictionary.word2idx[self.data[idx]]
            else:
                return self.dictionary.word2idx['OOV']

def collate_fn(batch):
    if isinstance(batch[0], tuple):
        return batch
    else:
        return torch.LongTensor(batch)

def create(datapath, dictfile, batchSize=1, shuffle=False, workers=0, sep=False):
    loaders = []
    dictionary = Dictionary(dictfile)
    for split in ['train', 'valid', 'test']:
        data_file = os.path.join(datapath, '%s.txt' %split)
        if split != 'train':
            dataset = LMdata(data_file, dictionary, sep)
        else:
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
