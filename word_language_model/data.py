import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def load(self, fname):
        with open(fname, encoding='utf-8') as fh:
            self.idx2word = [word.strip() for word in fh]
        self.word2idx = {word:idx for idx,word in enumerate(self.idx2word)}

class Corpus(object):
    def __init__(self, path, vocab_file=None, train_file='train.txt', valid_file='valid.txt', test_file='test.txt'):
        self.keep_unknown = vocab_file is None

        self.dictionary = Dictionary()
        if vocab_file is not None:
            self.dictionary.load(vocab_file)

        if path is not None:
            self.train = self.tokenize(os.path.join(path, train_file))
            self.valid = self.tokenize(os.path.join(path, valid_file))
            self.test = self.tokenize(os.path.join(path, test_file))

    @classmethod
    def from_vocab(cls, vocab_file):
        return cls(None, vocab_file=vocab_file)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        if self.keep_unknown:
            with open(path, 'r', encoding="utf8") as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
                    else:
                        ids[token] = self.dictionary.word2idx['<unk>']
                    token += 1

        return ids

    def print_vocab_file(self, path):
        with open(path, 'w', encoding='utf8') as f:
            for word in self.idx2word:
                print(word, file=path)
