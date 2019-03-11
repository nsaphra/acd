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
parser.add_argument('--important_score_file', type=str, default='importance.csv',
                    help='path to the file to save important score information')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
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
ntokens = len(corpus.dictionary)
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
    else:
        return tuple(repackage_hidden(v) for v in h)

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

def new_word_importance_dict():
    return {"relevant_word":[], "target_word":[], "relevant_position":[], "target_position":[], "relevant_norm":[], "irrelevant_norm":[], "relevant_target_score":[], "irrelevant_target_score":[]}

#   TODO we can increase efficiency by only multiplying the vector after the stop point
def propagate_softmax(softmax_layer, relevant, irrelevant):
    W_out = softmax_layer.weight.cpu().numpy().transpose()
    relevant_scores = np.dot(relevant, W_out)
    irrelevant_scores = np.dot(irrelevant, W_out)
    return relevant_scores, irrelevant_scores

def word_component_top(layer, softmax_layer, input_vector, data, targets, word_importance_lists, cell_state):
    relevant_words = []
    target_words = []
    target_positions = []
    relevant_positions = []
    relevant_target_scores = []
    irrelevant_target_scores = []
    relevant_norms = []
    irrelevant_norms = []
    for j, relevant_word in enumerate(data.cpu().numpy()):
        relevant, irrelevant = cd.cd_lstm(layer, input_vector, start = j, stop = j, cell_state=cell_state)
        relevant_scores, irrelevant_scores = propagate_softmax(softmax_layer, relevant, irrelevant)

        for i, target_word in enumerate(targets[j:], j):
            relevant_words.append(relevant_word)
            target_words.append(target_word)
            relevant_positions.append(j)
            target_positions.append(i+1)

            relevant_target_scores.append(relevant_scores[j, target_word])
            irrelevant_target_scores.append(irrelevant_scores[j, target_word])
            relevant_norms.append(np.linalg.norm(relevant_scores[j]))
            irrelevant_norms.append(np.linalg.norm(irrelevant_scores[j]))

    word_importance_lists["relevant_word"] += relevant_words
    word_importance_lists["target_word"] += target_words
    word_importance_lists["relevant_position"] += relevant_positions
    word_importance_lists["target_position"] += target_positions
    word_importance_lists["relevant_norm"] += relevant_norms
    word_importance_lists["irrelevant_norm"] += irrelevant_norms
    word_importance_lists["relevant_target_score"] += relevant_target_scores
    word_importance_lists["irrelevant_target_score"] += irrelevant_target_scores

def evaluate_lstm_top(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = len(corpus.dictionary)
    importance_file = open(args.important_score_file, 'w')
    df = DataFrame.from_dict(new_word_importance_dict())
    df.to_csv(importance_file)

    word_importance_lists = new_word_importance_dict()
    hidden = model.init_hidden(args.eval_batch_size)

    decomposed_layer_number = model.nlayers-1
    decomposed_layer = getattr(model, model.rnn_module_name(decomposed_layer_number))
    with torch.no_grad():
        for batch, i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
            data, targets = get_batch(data_source, i)

            output = model.encoder(data)
            for layer in range(decomposed_layer_number):
                output, hidden[layer] = getattr(self, model.rnn_module_name(layer))(output, hidden[layer])

            word_component_top(decomposed_layer, model.decoder, output, data, targets, word_importance_lists, hidden[decomposed_layer_number])

            # Sanity check: is this output the same as the approximated output?
            # update hidden state
            output, hidden[decomposed_layer_number] = decomposed_layer(output, hidden[decomposed_layer_number])
            hidden = repackage_hidden(hidden)

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f}'.format(
                    epoch, batch, len(test_data) // args.bptt,
                    elapsed * 1000 / args.log_interval))
                total_loss = 0
                start_time = time.time()
        df.to_csv(importance_file)
    return DataFrame.from_dict(word_importance_lists)

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

with open(args.saved_model, 'rb') as f:
    model = torch.load(f)
    model.to(device)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.flatten_parameters()
    print("Model loaded.")

# Run on test data.
test_loss = evaluate_lstm_top(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
