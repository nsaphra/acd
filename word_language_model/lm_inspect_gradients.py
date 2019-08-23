# -*- coding: <utf-8> -*-

import sys
import os
import numpy
import argparse
import time
import math
import torch
import torch.nn as nn
from random import shuffle
import pandas
from scipy import sparse
import pickle
import ast
import json
import torch.nn.functional as F

import data
from gradient_analyzer import GradientAnalyzer

parser = argparse.ArgumentParser(description='Evaluate language model semantic and syntactic features')

# Model parameters.
parser.add_argument('--data', type=str, default=None,
                    help='location of the data corpus')
parser.add_argument('--saved_model', type=str,
                    help='model saved_model to use')
parser.add_argument('--gradient-outf', type=str, default=None,
                    help='output file for analysis on gradients')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--log-interval', type=int, default=200)
parser.add_argument('--test_corpus_file', type=str, default='test.txt',
                    help='path to the test corpus in the data directory')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
torch.set_default_dtype(torch.float64)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
first_symbol = corpus.dictionary.word2idx['(']
final_symbol = corpus.dictionary.word2idx[')']
conduit_length = list(corpus.test.cpu().numpy()).index(final_symbol) - list(corpus.test.cpu().numpy()).index(first_symbol) - 1
print("Data loaded.")

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    elif isinstance(h, list):
        return [repackage_hidden(v) for v in h]
    elif isinstance(h, tuple):
        return tuple(repackage_hidden(v) for v in h)
    else:
        raise TypeError()

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_rule_boundaries(data):
    # remove parenthetical symbols that cross a bptt boundary
    start_indices = [i for i,x in enumerate(data) if x == first_symbol]
    stop_indices = [i for i,x in enumerate(data) if x == final_symbol]
    if len(stop_indices) == 0 or len(start_indices) == 0:
        return None, None
    if start_indices[0] > stop_indices[0]:
        start_indices = start_indices[1:] # the sequence starts in the middle of a dependency
        if len(start_indices) == 0:
            return None, None
    if start_indices[-1] > stop_indices[-1]:
        stop_indices = stop_indices[:-1]
        if len(stop_indices) == 0:
            return None, None
    assert(len(start_indices) == len(stop_indices))
    return start_indices, stop_indices
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

print('Batching eval text data.')
batch_size = 1
test_data = batchify(corpus.tokenize(os.path.join(args.data, args.test_corpus_file)), batch_size)

criterion = nn.CrossEntropyLoss()
softmax = torch.nn.LogSoftmax(dim=1)

def evaluate(data_source):
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    num_batches = len(data_source) // args.bptt

    analyzer = GradientAnalyzer(model, l1=True, l2=True, mean=True, magnitude=True, range=True, variance=True, concentration=True)
    analyzer.add_hooks_to_model()

    model.train()
    for batch, i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
        data, targets = get_batch(data_source, i)
        analyzer.set_word_sequence(data, get_rule_boundaries(data))

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)
        loss.backward()

        total_scores = softmax(output_flat)
        analyzer.store_external_stats('final_symbol_score', total_scores[:, final_symbol])

        hidden = repackage_hidden(hidden)

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | ms/batch {:5.2f} |'.format(
                    batch, num_batches, elapsed * 1000 / args.log_interval))

            start_time = time.time()

    analyzer.compute_and_clear(args.gradient_outf)

with open(args.saved_model, 'rb') as f:
    model = torch.load(f)
    model.to(device)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.flatten_parameters()
    print("Model loaded.")

# Run on test data.
evaluate(test_data)
