import torch
import os
from random import seed, shuffle
from operator import itemgetter

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
    def add_vocab(self, path):
        with open(path, 'r') as f:
            sentence_list = []
            for line in f:
                words = line.split() + ['</s>']
                sentence_list.append((words, len(words)))
                for word in words:
                    self.dictionary.add_word(word)
        return sentence_list

    def arrange_bunch(self, path, batchsize):
        sentence_list = self.add_vocab(path)
        sentence_list.sort(key=itemgetter(1))
        utt_num = len(sentence_list)
        batch_num = utt_num // batchsize
        shuffle_step = 5
        all_data_batches = []
        for i in range(0, batch_num-shuffle_step, shuffle_step):
            shuffle_unit = sentence_list[i*batchsize:(i+5)*batchsize]
            seed(i)
            shuffle(shuffle_unit)
            for j in range(shuffle_step):
                sentence_batch = shuffle_unit[j*batchsize:(j+1)*batchsize]
                maxLen = max(sentence_batch, key=itemgetter(1))[1]
                ids = torch.zeros([batchsize, maxLen], dtype=torch.int64)
                for s_ind, sentence in enumerate(sentence_batch):
                    for w_ind, word in enumerate(sentence[0]):
                        ids[s_ind][w_ind] = self.dictionary.word2idx[word]
                all_data_batches.append(ids)
        return all_data_batches

    def get_data_eval(self, path):
        sentence_list = self.add_vocab(path)
        utt_list = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['</s>']
                ids = torch.zeros([len(words), 1], dtype=torch.int64)
                for ind, word in enumerate(words):
                    ids[ind] = self.dictionary.word2idx[word]
                utt_list.append(ids.view(1, -1))
        return utt_list

