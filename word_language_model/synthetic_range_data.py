# coding: utf-8
"""
In these experiments, we are synthesizing data that has long distant rules of some particular length
surrounding an inner attractor which may or may not appear a number of times elsewhere in the corpus.
The goal here is to see how the length of the dependency  affects model capacity to learn the longer rule,
and to see how familiarity with the  attractor affects the capacity as well.
"""
import argparse
import time
import math
import os
import random
import sys

parser = argparse.ArgumentParser(description='Generate synthetic data for learning specific long-range rules')
parser.add_argument('--outdir', type=str, default=None,
                    help='location to put the data corpus')
parser.add_argument('--rule_length', type=int, default=10,
                    help='number of tokens between the matched pair')
parser.add_argument('--corpus_length', type=int, default=1000000)
parser.add_argument('--vocab_size', type=int, default=1000)
parser.add_argument('--rule_count', type=int, default=1000)
parser.add_argument('--attractor_count', type=int, default=1)
parser.add_argument('--test_corpus_length', type=int, default=50000)
parser.add_argument('--sequence_length', type=int, default=35)
parser.add_argument('--conduit_vocab_size', type=int, default=100)
args = parser.parse_args()

if args.outdir is None:
    # args.outdir = 'data/synthetic/vocab{}_corpuslength{}_rulecount{}_rulelength{}_attractorcount{}'.format(args.vocab_size, args.corpus_length, args.rule_count, args.rule_length, args.attractor_count)
    args.outdir = '/disk/ostrom/s1477768/synthetic_context/data/vocab{}_corpuslength{}_rulecount{}_rulelength{}_attractorcount{}_conduitvocab{}'.format(args.vocab_size, args.corpus_length, args.rule_count, args.rule_length, args.attractor_count, args.conduit_vocab_size)
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
train = open(args.outdir+'/train.txt', 'w')
valid = open(args.outdir+'/valid.txt', 'w')
test = open(args.outdir+'/test.txt', 'w')
domain_test = open(args.outdir+'/domain_test.txt', 'w')

document = [random.randint(0, args.vocab_size) for x in range(args.corpus_length)]

conduit_vocab = [[random.randint(0, args.vocab_size) for x in range(args.rule_length)] for x in range(args.conduit_vocab_size)]


rule_locations = random.sample(range(0, len(document), args.rule_length), args.attractor_count * args.rule_count)
for begin_idx in range(0, len(rule_locations), args.attractor_count):
    conduit = conduit_vocab[begin_idx // args.attractor_count % len(conduit_vocab)]
    begin = rule_locations[begin_idx]
    document[begin] = '('
    document[begin + 1:begin + args.rule_length - 1] = conduit
    document[begin + args.rule_length - 1] = ')'
    for attractor_copy_begin in rule_locations[begin_idx+1:begin_idx+args.attractor_count]:
        document[attractor_copy_begin:attractor_copy_begin+args.rule_length-1] = conduit

print(' '.join([str(x) for x in document]), file=train)
train.close()

valid_document = [random.randint(0, args.vocab_size) for x in range(args.test_corpus_length)]
test_document = [random.randint(0, args.vocab_size) for x in range(args.test_corpus_length)]
domain_test_document = [random.randint(0, args.vocab_size) for x in range(args.test_corpus_length)]
rule_locations = [random.randint(0, args.sequence_length - args.rule_length) for x in range(args.test_corpus_length//args.sequence_length)]
for begin_idx in range(0, len(rule_locations)):
    begin = begin_idx * args.sequence_length + rule_locations[begin_idx]
    valid_document[begin] = '('
    valid_document[begin + args.rule_length - 1] = ')'
    test_document[begin] = '('
    test_document[begin + args.rule_length - 1] = ')'
    domain_test_document[begin] = '('
    domain_test_document[begin + 1:begin + args.rule_length - 1] = conduit_vocab[begin_idx // args.attractor_count % len(conduit_vocab)]
    domain_test_document[begin + args.rule_length - 1] = ')'

print(' '.join([str(x) for x in test_document]), file=test)
print(' '.join([str(x) for x in domain_test_document]), file=domain_test)
print(' '.join([str(x) for x in valid_document]), file=valid)
valid.close()
test.close()
