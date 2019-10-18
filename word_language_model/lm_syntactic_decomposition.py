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
import ud_util

sys.path.append('..')
import acd.scores.cd as cd

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--layer_number', type=int, default=0, metavar='N',
                    help='index of the LSTM layer to decompose')
parser.add_argument('--saved_model', type=str, default='model.pt',
                    help='path to the model')
parser.add_argument('--vocab_file', type=str, default='./data/wikitext-2/vocab.txt',
                    help='location of the vocab list')
parser.add_argument('--conll_file', type=str, default='test.conll',
                    help='path to the test corpus in the data directory')
parser.add_argument('--score_file', type=str, default='importance.csv',
                    help='path to the file to save score information')
parser.add_argument('--left_pos', type=str, default=None,
                    help='part of speech of the left word in the pair')
parser.add_argument('--right_pos', type=str, default=None,
                    help='part of speech of the right word in the pair')
parser.add_argument('--max_pair_distance', type=int, default=5,
                    help='random seed')
parser.add_argument('--target_offset', type=int, default=0,
                    help='random seed')
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

dictionary = data.Dictionary.from_file(args.vocab_file)
corpus = ud_util.ConllDocument(args.conll_file, dictionary.word2idx)
print("Data loaded.")

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

class PairProperties():
    def __init__(self, decomposer):
        self.decomposer = decomposer
        self.clear()

    def update(self, sentence, left, right, target):
        self.left_pos_tag.append(sentence[left].universal_pos)
        self.right_pos_tag.append(sentence[right].universal_pos)
        self.target_pos_tag.append(sentence[target].universal_pos)
        self.left_idx.append(left)
        self.right_idx.append(right)
        self.target_idx.append(target)
        self.tree_relation.append(sentence.get_tree_relation(left, right))
        self.sentence_text.append(str(sentence))

    def clear(self):
        self.left_pos_tag = []
        self.right_pos_tag = []
        self.target_pos_tag = []
        self.left_idx = []
        self.right_idx = []
        self.target_idx = []
        self.tree_relation = []
        self.sentence_text = []

    def to_dict(self):
        return {
            "left_pos_tag": self.left_pos_tag,
            "right_pos_tag": self.right_pos_tag,
            "target_pos_tag": self.target_pos_tag,
            "left_idx": self.left_idx,
            "right_idx": self.right_idx,
            "target_idx": self.target_idx,
            "tree_relation": self.tree_relation,
            "sentence_text": self.sentence_text,
        }

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
        final_symbol = self.decomposer.targets[final_idx, batch]

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

        self.relevant_input_norm.append((relevant_inputs[final_idx, batch]).norm().cpu().numpy())
        self.irrelevant_input_norm.append((irrelevant_inputs[final_idx, batch]).norm().cpu().numpy())
        self.relevant_irrelevant_diff_input_norm.append((relevant_inputs[final_idx, batch] - irrelevant_inputs[final_idx, batch]).norm().cpu().numpy())
        self.relevant_irrelevant_input_ratio.append((relevant_inputs[final_idx, batch].norm() / irrelevant_inputs[final_idx, batch].norm()).cpu().numpy())

        self.relevant_hidden_norm.append((relevant[final_idx, batch]).norm().cpu().numpy())
        self.irrelevant_hidden_norm.append((irrelevant[final_idx, batch]).norm().cpu().numpy())
        self.relevant_irrelevant_diff_hidden_norm.append((relevant[final_idx, batch] - irrelevant[final_idx, batch]).norm().cpu().numpy())
        self.relevant_irrelevant_hidden_ratio.append((relevant[final_idx, batch].norm() / irrelevant[final_idx, batch].norm()).cpu().numpy())

    def update_approximation_error(self, relevant, irrelevant, true_output, final_idx):
        batch = 0

        self.approximate_output_norm.append((relevant[final_idx,batch] + irrelevant[final_idx,batch] + model.decoder.bias).norm().cpu().numpy())
        self.approximation_error_norm.append((relevant[final_idx,batch] + irrelevant[final_idx,batch] + model.decoder.bias - true_output[final_idx,batch]).norm().cpu().numpy())

    def to_dict(self, prefix=''):
        if prefix != '':
            prefix = prefix+'_'
        return {
            prefix+"approximate_output_norm":self.approximate_output_norm,
            prefix+"approximation_error_norm":self.approximation_error_norm,

            prefix+"relevant_target_score":self.relevant_target_scores,
            prefix+"irrelevant_target_score":self.irrelevant_target_scores,
            prefix+"total_target_score":self.total_target_scores,
            prefix+"importance":self.importances,

            prefix+"irrelevant_input_norm":self.relevant_input_norm,
            prefix+"irrelevant_input_norm":self.irrelevant_input_norm,
            prefix+"relevant_irrelevant_input_ratio":self.relevant_irrelevant_input_ratio,
            prefix+"relevant_irrelevant_diff_input_norm":self.relevant_irrelevant_diff_input_norm,

            prefix+"relevant_hidden_norm":self.relevant_hidden_norm,
            prefix+"irrelevant_hidden_norm":self.irrelevant_hidden_norm,
            prefix+"relevant_irrelevant_hidden_ratio":self.relevant_irrelevant_hidden_ratio,
            prefix+"relevant_irrelevant_diff_hidden_norm":self.relevant_irrelevant_diff_hidden_norm,

        }

    def clear_scores(self):
        # relevant: bptt x hidden, first index is target word
        self.approximate_output_norm = []
        self.approximation_error_norm = []

        self.relevant_target_scores = []
        self.irrelevant_target_scores = []
        self.total_target_scores = []
        self.importances = []

        self.relevant_input_norm = []
        self.irrelevant_input_norm = []
        self.relevant_irrelevant_input_ratio = []
        self.relevant_irrelevant_diff_input_norm = []

        self.relevant_hidden_norm = []
        self.irrelevant_hidden_norm = []
        self.relevant_irrelevant_hidden_ratio = []
        self.relevant_irrelevant_diff_hidden_norm = []

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

    def decompose_layer(self, output, relevant_list):
        decomposed_layer = getattr(self.model, self.model.rnn_module_name(self.decomposed_layer_number))
        relevant, irrelevant = cd.cd_lstm(decomposed_layer, output, relevant_list=relevant_list, cell_state=self.hidden[self.decomposed_layer_number])

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

def pos_constrained_interaction_sets(sentence_conll, left_pos, right_pos, internal_offset, target_offset=0):
    interaction_sets = []
    for i, token in enumerate(sentence_conll[:-internal_offset-target_offset]):
        right_token_idx = i+internal_offset
        if token.u_pos == left_pos and sentence_conll[right_token_idx] == right_pos:
            yield (i, right_token_idx, right_token_idx + target_offset)

def positive_offset(negative_offset):
    # with negative_offset=0, we are actually looking for the last item
    # in the conduit, so if the conduit is length 8, we want conduit[7]
    return conduit_length - negative_offset - 1

def get_interaction_sets(sentence):
    last_right = len(sentence) - 2 - args.target_offset
    for left in range(last_right - 1):
        for pair_distance in range(1, min(args.max_pair_distance, last_right - left)):
            right = left+pair_distance
            target = right+args.target_offset

            if args.left_pos is not None and args.right_pos is not None:
                if sentence[left].universal_pos != args.left_pos \
                        or sentence[right].universal_pos != args.right_pos:
                    continue
            yield (left, right, target)

def evaluate_lstm(data_source, decomposed_layer_number):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    hidden = model.init_hidden(1)
    decomposer = ModelDecomposer(model, hidden, decomposed_layer_number)
    nonlinearity_scores = ImportanceScores(decomposer) # nonlinear interactions between the open symbol and the conduit
    pair_properties = PairProperties(decomposer)
    true_output_norm = []

    def clear_scores():
        true_output_norm.clear()
        nonlinearity_scores.clear_scores()
        pair_properties.clear()

    def score_dataframe():
        scores = {"true_output_norm":true_output_norm}
        scores.update(nonlinearity_scores.to_dict(prefix='nonlinearity'))
        scores.update(pair_properties.to_dict())
        return DataFrame.from_dict(scores)

    score_file = open(args.score_file, 'w')
    print('Printing importance information to {}'.format(args.score_file))
    score_dataframe().to_csv(score_file)

    start_time = time.time()
    with torch.no_grad():
        for sentence_idx,sentence in enumerate(data_source.sentences()):
            data, targets = data_source.vectorize(sentence)
            data = data.to(device)
            targets = targets.to(device)
            if len(data) == 0:
                continue
            decomposer.set_data_batch(data, targets)

            lower_output = decomposer.run_lower_layers()

            for (left, right, target) in get_interaction_sets(sentence):
                true_output = decomposer.rerun_layers(lower_output, update_hidden=False)
                # parallel batch index is always 0, because the batch size is always 1
                true_output_norm.append(true_output[target,0].norm().cpu().numpy())

                pair_relevant, pair_irrelevant = decomposer.decompose_layer(lower_output[:target+1], [left, right])
                left_relevant, left_irrelevant = decomposer.decompose_layer(lower_output[:target+1], [left])
                right_relevant, right_irrelevant = decomposer.decompose_layer(lower_output[:target+1], [right])
                nonlinearity_scores.update_from_decomposition(pair_relevant - left_relevant - right_relevant, pair_relevant, target, true_output)

                pair_properties.update(sentence, left, right, target)

            # update hidden state
            # output = decomposer.rerun_layers(lower_output)
            decomposer.hidden = repackage_hidden(decomposer.hidden)

            if sentence_idx % args.log_interval == 0 and sentence_idx > 0:
                elapsed = time.time() - start_time
                print('| {:5d} sentences | ms/sentence {:5.2f}'.format(
                    sentence_idx,
                    elapsed * 1000 / args.log_interval))
                start_time = time.time()

                score_dataframe().to_csv(score_file, header=False)
                clear_scores()
                score_file.flush()

with open(args.saved_model, 'rb') as f:
    model = torch.load(f)
    model.to(device)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.flatten_parameters()
    print("Model loaded.")

# Run on test data.
evaluate_lstm(corpus, args.layer_number)
