# coding: utf-8
import argparse
import time
import math
import os
from pandas import DataFrame
import sys
import torch
import torch.nn as nn
import torch.onnx
import numpy as np

import data
import model

sys.path.append('..')
import acd.scores.cd as cd

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--eval_batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--saved_model', type=str, default='model.pt',
                    help='path to the model')
parser.add_argument('--importance_score_file', type=str, default='importance.csv',
                    help='path to the file to save important score information')
parser.add_argument('--layer_number', type=int, default=0, metavar='N',
                    help='index of the LSTM layer to decompose')
parser.add_argument('--calculate_incremental_effects', action='store_true',
                    help='calculate target probability at timestep in the conduit')
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

test_data = batchify(corpus.test, args.eval_batch_size)
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

class ImportanceScores():
    def __init__(self, decomposer):
        self.decomposer = decomposer
        self.softmax = torch.nn.LogSoftmax(dim=2)

        self.clear_scores()

    def update_from_decomposition(self, relevant, irrelevant, stop_idx, true_output):
        relevant_scores, irrelevant_scores = self.decomposer.run_upper_layers(relevant, irrelevant)
        self.update_metrics(relevant, irrelevant, relevant_scores, irrelevant_scores, stop_idx)

        self.update_approximation_error(relevant_scores, irrelevant_scores, true_output, stop_idx)

    def update_metrics(self, relevant, irrelevant, relevant_inputs, irrelevant_inputs, final_idx):
        batch = 0

        relevant_scores = self.softmax(relevant_inputs)
        irrelevant_scores = self.softmax(irrelevant_inputs)
        total_inputs = relevant_inputs + irrelevant_inputs
        total_scores = self.softmax(total_inputs)
        input_difference = relevant_inputs - irrelevant_inputs
        input_difference_scores = self.softmax(input_difference)

        self.relevant_target_scores.append(relevant_scores[final_idx, batch, final_symbol].cpu().numpy())
        self.irrelevant_target_scores.append(irrelevant_scores[final_idx, batch, final_symbol].cpu().numpy())
        self.total_target_scores.append(total_scores[final_idx, batch, final_symbol].cpu().numpy())

        self.importances.append(input_difference_scores[final_idx, batch, final_symbol].cpu().numpy())
        self.relevant_input_score_norm.append((relevant_inputs[final_idx, batch] - irrelevant_inputs[final_idx, batch]).norm().cpu().numpy())
        self.relevant_irrelevant_input_ratio.append((relevant_inputs[final_idx, batch].norm() / irrelevant_inputs[final_idx, batch].norm()).cpu().numpy())

    def update_approximation_error(self, relevant, irrelevant, true_output, final_idx):
        batch = 0

        self.approximate_output_norm.append((relevant[final_idx,batch] + irrelevant[final_idx,batch] + model.decoder.bias).norm().cpu().numpy())
        self.approximation_error_norm.append((relevant[final_idx,batch] + irrelevant[final_idx,batch] + model.decoder.bias - true_output[final_idx,batch]).norm().cpu().numpy())

    def to_dict(self, prefix=''):
        if prefix is not '':
            prefix = prefix+'_'
        return {
            prefix+"relevant_target_score":self.relevant_target_scores,
            prefix+"irrelevant_target_score":self.irrelevant_target_scores,
            prefix+"total_target_score":self.total_target_scores,
            prefix+"relevant_input_score_norm":self.relevant_input_score_norm,
            prefix+"importance":self.importances,
            prefix+"approximate_output_norm":self.approximate_output_norm,
            prefix+"approximation_error_norm":self.approximation_error_norm,
            prefix+"relevant_irrelevant_input_ratio":self.relevant_irrelevant_input_ratio,
        }

    def clear_scores(self):
        # relevant: bptt x hidden, first index is target word
        self.relevant_target_scores = []
        self.irrelevant_target_scores = []
        self.total_target_scores = []
        self.relevant_input_score_norm = []
        self.importances = []
        self.approximate_output_norm = []
        self.approximation_error_norm = []
        self.relevant_irrelevant_input_ratio = []

class ModelDecomposer():
    def __init__(self, model, hidden, decomposed_layer_number):
        self.model = model
        self.hidden = hidden
        self.softmax_layer = model.decoder.weight.t()
        self.decomposed_layer_number = decomposed_layer_number

    def set_data_batch(self, data, targets):
        self.data = data
        self.targets = targets.view(data.shape)

    #   TODO we can increase efficiency by only multiplying the vector after the stop point
    # relevant, irrelevant: bptt x hidden
    # softmax_layer (weight matrix transpose): hidden x vocab_size
    def through_softmax(self, relevant, irrelevant):
        relevant_scores = torch.matmul(relevant, self.softmax_layer)
        irrelevant_scores = torch.matmul(irrelevant, self.softmax_layer)
        return relevant_scores, irrelevant_scores

    def decompose_layer(self, output, start, stop):
        decomposed_layer = getattr(self.model, self.model.rnn_module_name(self.decomposed_layer_number))
        relevant, irrelevant = cd.cd_lstm(decomposed_layer, output, start = start, stop = stop, cell_state=self.hidden[self.decomposed_layer_number])

        # sanity check:
        # true_output, true_hidden = decomposed_layer(output, self.hidden[self.decomposed_layer_number])
        # print((relevant[-1] + irrelevant[-1] - true_output[-1]).norm().cpu().numpy())
        # TODO integrate sanity check into overall code and throw out bad examples

        return relevant, irrelevant

    def run_lower_layers(self):
        output = model.encoder(self.data)
        for layer in range(self.decomposed_layer_number):
            output, self.hidden[layer] = getattr(self.model, self.model.rnn_module_name(layer))(output, self.hidden[layer])
        return output

    def run_upper_layers(self, relevant, irrelevant):
        for layer in range(self.decomposed_layer_number+1, self.model.nlayers):
            relevant, irrelevant = cd.through_lstm(getattr(self.model, self.model.rnn_module_name(layer)), relevant, irrelevant, self.hidden[layer])
        return self.through_softmax(relevant, irrelevant)

    def rerun_layers(self, output, update_hidden=True):
        for layer in range(self.decomposed_layer_number, self.model.nlayers):
            output, _ = getattr(self.model, self.model.rnn_module_name(layer))(output, self.hidden[layer])
            if update_hidden:
                self.hidden[layer] = _
        return model.decoder(output)

def evaluate_lstm(data_source, decomposed_layer_number):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    hidden = model.init_hidden(args.eval_batch_size)
    decomposer = ModelDecomposer(model, hidden, decomposed_layer_number)
    symbol_scores = ImportanceScores(decomposer)
    conduit_scores = ImportanceScores(decomposer)
    if args.calculate_incremental_effects:
        incremental_scores = [ImportanceScores(decomposer) for x in range(conduit_length+1)]
    true_output_norm = []

    def clear_scores():
        symbol_scores.clear_scores()
        conduit_scores.clear_scores()
        true_output_norm.clear()

        if args.calculate_incremental_effects:
            for inc, inc_scores in enumerate(incremental_scores):
                inc_scores.clear_scores()

    def score_dataframe():
        scores = {"true_output_norm":true_output_norm}
        scores.update(symbol_scores.to_dict())
        scores.update(conduit_scores.to_dict(prefix='conduit'))

        if args.calculate_incremental_effects:
            for inc, inc_scores in enumerate(incremental_scores):
                scores.update(inc_scores.to_dict(prefix=str(inc)))

        return DataFrame.from_dict(scores)

    importance_file = open(args.importance_score_file, 'w')
    print('Printing importance information to {}'.format(args.importance_score_file))
    score_dataframe().to_csv(importance_file)

    start_time = time.time()
    with torch.no_grad():
        for batch, i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
            data, targets = get_batch(data_source, i)
            decomposer.set_data_batch(data, targets)

            lower_output = decomposer.run_lower_layers()

            #TODO: align the long-distance dependencies so that every sequence has one
            # remove parenthetical symbols that cross a bptt boundary
            start_indices = [i for i,x in enumerate(data) if x[0] == first_symbol]
            stop_indices = [i for i,x in enumerate(data) if x[0] == final_symbol]
            if len(stop_indices) == 0 or len(start_indices) == 0:
                continue
            if start_indices[0] > stop_indices[0]:
                start_indices = start_indices[1:] # the sequence starts in the middle of a dependency
                if len(start_indices) == 0:
                    continue
            if start_indices[-1] > stop_indices[-1]:
                stop_indices = stop_indices[:-1]
                if len(stop_indices) == 0:
                    continue
            assert(len(start_indices) == len(stop_indices))

            for i in range(len(start_indices)):
                stop_idx = stop_indices[i]
                start_idx = start_indices[i]

                true_output = decomposer.rerun_layers(lower_output, update_hidden=False)
                # parallel batch index is always 0, because the batch size is always 1
                true_output_norm.append(true_output[stop_idx,0].norm().cpu().numpy())

                relevant, irrelevant = decomposer.decompose_layer(lower_output, start_idx, start_idx)
                symbol_scores.update_from_decomposition(relevant, irrelevant, stop_idx, true_output)

                if stop_idx - start_idx == 0:
                    continue

                conduit_relevant, conduit_irrelevant = decomposer.decompose_layer(lower_output, start_idx+1, stop_idx-1)
                conduit_scores.update_from_decomposition(conduit_relevant, conduit_irrelevant, stop_idx, true_output)

                if args.calculate_incremental_effects:
                    for inc, scores in enumerate(incremental_scores):
                        inc_relevant, inc_irrelevant = decomposer.decompose_layer(lower_output, start_idx, start_idx+inc)
                        scores.update_from_decomposition(inc_relevant, inc_irrelevant, stop_idx, true_output)

                # sanity check:
                # print((relevant_scores[-1] + irrelevant_scores[-1] + model.decoder.bias).norm().cpu().numpy())
                # print(decomposer.rerun_layers(lower_output, update_hidden=False)[-1].norm().cpu().numpy())
                # print((relevant_scores[-1] + irrelevant_scores[-1] + model.decoder.bias - decomposer.rerun_layers(lower_output, update_hidden=False)[-1]).norm().cpu().numpy())

            # update hidden state
            output = decomposer.rerun_layers(lower_output)
            decomposer.hidden = repackage_hidden(decomposer.hidden)

            if batch % args.log_interval == 0 and batch > 0:
                elapsed = time.time() - start_time
                print('| {:5d}/{:5d} batches | ms/batch {:5.2f}'.format(
                    batch, len(test_data) // args.bptt,
                    elapsed * 1000 / args.log_interval))
                start_time = time.time()

                score_dataframe().to_csv(importance_file, header=False)
                clear_scores()
                importance_file.flush()

with open(args.saved_model, 'rb') as f:
    model = torch.load(f)
    model.to(device)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.flatten_parameters()
    print("Model loaded.")

# Run on test data.
evaluate_lstm(test_data, args.layer_number)
