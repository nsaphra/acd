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
from collections import namedtuple

sys.path.append( 'torch_example/')
sys.path.append( 'acd')

import data
import acd.scores.cd
parser = argparse.ArgumentParser(description='Explore word importance of language model in terms of dependency lengths')

# Model parameters.
parser.add_argument('--data', type=str, default=None,
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint to use')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--vocab-list', type=str,
                    help='vocab file')
parser.add_argument('--test-data', type=str,
                    help='location of the valid and test data corpus')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--log-interval', type=int, default=200)
parser.add_argument('--save-file', type=str, default=None)
parser.add_argument('--test-length', type=int, default=None,
                    help='number of lines to test in the test corpus')

args = parser.parse_args()


batch_size = 1

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda" if args.cuda else "cpu")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)

corpus = data.Corpus(args.data, args.test_data, vocab_file=args.vocab_list, use_curriculum=False, test_length=args.test_length)

def batchify(data, bsz):
    print('batching data ...', end='')
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    return data.to(device)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


print('Batching eval text data.', file=sys.stderr)
test_data = batchify(corpus.test, batch_size)
ntokens = len(corpus.dictionary)

criterion = nn.CrossEntropyLoss()

def new_word_importance_dict():
    return {"relevant_word":[], "target_word":[], "relevant_position":[], "target_position":[], "relevant_importance":[], "irrelevant_importance":[]}

def apply_softmax_cd(softmax_layer, relevant, irrelevant):
    relevant_scores = np.dot(W_out, relevant_h[-1])
    irrelevant_scores = np.dot(W_out, irrelevant_h[-1])
    return relevant_scores, irrelevant_scores

def word_component_top(layer, softmax_layer, input_vector, data, targets, word_importance_lists, cell_state):
    word = targets[-1]
    target_position_in_input_data = len(targets)
    relevant_words = data.data[:target_position_in_input_data]
    relevant_positions = list(range(target_position_in_input_data))
    relevant_target_scores = []
    irrelevant_target_scores = []
    relevant_norms = []
    irrelevant_norms = []
    for j in relevant_positions:
        relevant, irrelevant = cd.cd_lstm(layer, input_vector[:target_position_in_input_data], start = j, stop = j)
        relevant.to(device)
        irrelevant.to(device)
        relevant_scores, irrelevant_scores = apply_softmax_cd(softmax_layer, relevant, irrelevant)
        relevant_target_scores += relevant_scores[word]
        irrelevant_target_scores += irrelevant_scores[word]
        relevant_norms += relevant_scores.norm(2)
        irrelevant_norms += irrelevant_scores.norm(2)


    word_importance_lists["relevant_word"] += relevant_words
    word_importance_lists["target_word"] += target_words
    word_importance_lists["relevant_position"] += relevant_positions
    word_importance_lists["target_position"] += target_positions
    word_importance_lists["relevant_norm"] += relevant_norms
    word_importance_lists["irrelevant_norm"] += irrelevant_norms
    word_importance_lists["relevant_target_score"] += relevant_target_scores
    word_importance_lists["irrelevant_target_score"] += irrelevant_target_scores

    return DataFrame.from_dict(word_importance_lists)


def word_component_bottom(layer, softmax_layer, input_vector, data, targets, word_importance_lists, cell_state):
    model.eval()
    word = targets[-1]
    target_position_in_input_data = len(targets)
    relevant_words = data.data[:target_position_in_input_data]
    relevant_positions = list(range(target_position_in_input_data))
    relevant_target_scores = []
    irrelevant_target_scores = []
    relevant_norms = []
    irrelevant_norms = []
    for j in relevant_positions:
        relevant, irrelevant = cd.cd_lstm(layer, input_vector[:target_position_in_input_data], start = j, stop = j, cell_state=cell_state)
        relevant, irrelevant = cd.propagate_upper_lstm(softmax_layer)

        relevant_scores, irrelevant_scores = apply_softmax_cd(softmax_layer, relevant)
        relevant_target_scores += relevant_scores[word]
        irrelevant_target_scores += irrelevant_scores[word]
        relevant_norms += relevant_scores.norm(2)
        irrelevant_norms += irrelevant_scores.norm(2)


    word_importance_lists["relevant_word"] += relevant_words
    word_importance_lists["target_word"] += target_words
    word_importance_lists["relevant_position"] += relevant_positions
    word_importance_lists["target_position"] += target_positions
    word_importance_lists["relevant_norm"] += relevant_norms
    word_importance_lists["irrelevant_norm"] += irrelevant_norms
    word_importance_lists["relevant_target_score"] += relevant_target_scores
    word_importance_lists["irrelevant_target_score"] += irrelevant_target_scores


# def word_component(layer, input_vector, data, targets):
#     word_importance_lists = new_word_importance_dict()
#
#     for i, word in enumerate(targets.data):
#         target_position_in_input_data = i+1
#         relevant_words = data.data[:target_position_in_input_data]
#         relevant_positions = list(range(target_position_in_input_data))
#         importances = [cd.cd_lstm(layer, input_vector[:target_position_in_input_data], start = j, stop = j)
#                           for j in relevant_position]
#         relevant, irrelevant = zip(*importances)
#         target_words = [word]*target_position_in_input_data
#         target_positions = [target_position_in_input_data]*target_position_in_input_data
#
#         word_importance_lists["relevant_word"] += relevant_words
#         word_importance_lists["target_word"] += target_words
#         word_importance_lists["relevant_position"] += relevant_positions
#         word_importance_lists["target_position"] += target_positions
#         word_importance_lists["relevant_component"] += relevant
#         word_importance_lists["irrelevant_component"] += irrelevant
#
#     return DataFrame.from_dict(word_importance_lists)

def evaluate_lstm_top(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = len(corpus.dictionary)
    importance_file = open('my_csv.csv', 'w')
    df = pandas.DataFrame.from_dict(new_word_importance_dict())
    df.to_csv(importance_file)

    total_loss = 0
    word_importance_lists = new_word_importance_dict()
    hidden = model.init_hidden(batch_size)

    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i)
        # output_, hidden_ = model(data, hidden)

        emb = model.encoder(data)
        output1, hidden1 = model.rnn1(emb, hidden[0])

        # line_component = word_component_top(model.rnn2, model.decoder, output1, data, targets, word_importance_lists, hidden[1])

        # output_flat = output.view(-1, ntokens)
        # total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    df.to_csv(importance_file)
    return DataFrame.from_dict(word_importance_lists)
    # return total_loss[0] / len(data_source)

# Run on test data.
print("test data size ", test_data.size(), file=sys.stderr)


test_stats = evaluate_lstm_top(test_data)
print('=' * 89, file=sys.stderr)
print('| End | dev ppl {:8.2f} | loss {:5.2f}'.format(
      math.exp(test_loss), test_loss), file=sys.stderr)
print('=' * 89, file=sys.stderr)

with open(args.save_file, 'w') as file:
    test_stats.to_csv(file)
