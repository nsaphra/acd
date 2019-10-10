# coding: utf-8
import sys
import os
from collections import namedtuple
from enum import Enum
import torch

class ConllToken():
    def __init__(self, line_text, phantom_tokens):
        self.is_phantom = False
        self.discard_sentence = False
        def parse_inflections(conll_inflection):
            # format is "Property=Value|Property=Value"
            properties = conll_inflection.split('|')
            for property_value in properties:
                if property_value == '_':
                    continue
                property, value = property_value.split('=')
                yield (property, value)

        line = line_text.strip().split()
        try:
            self.id = int(line[0])
        except:
            # line[0] is a trace
            dependency = {property:value for property, value in parse_inflections(line[9])}
            if 'CopyOf' not in dependency or int(dependency['CopyOf']) < 0:
                print('sentence has uninterpretable trace: ', line_text)
                self.discard_sentence = True
            else:
                phantom_tokens[line[0]] = int(dependency['CopyOf'])
            self.is_phantom = True
            return

        self.form = line[1]
        self.lemma = line[2]
        self.universal_pos = line[3]
        self.x_pos = line[4]

        self.inflection = {property:value for property, value in parse_inflections(line[5])}

        try:
            self.head_idx = int(line[6])
        except:
            self.head_idx = phantom_tokens[line[6]]
        self.dependency_relation = line[7]

class TreeRelation(Enum):
    LEFT_HEAD = 1
    RIGHT_HEAD = 2
    LEFT_ANCESTOR = 3
    RIGHT_ANCESTOR = 4
    DISCONNECTED = 5

class ConllSentence():
    def __init__(self, file_handle):
        def should_ignore_line(text):
            return text.strip() == '' or text.startswith('#')

        def skip_sentence():
            self.tokens = []
            line = file_handle.readline()
            while line and (not should_ignore_line(line)):
                line = file_handle.readline()

        # parse next sentence from file
        self.tokens = []
        def read_next_sentence():
            line = file_handle.readline()
            while line and (should_ignore_line(line)):
                line = file_handle.readline()

            phantom_tokens = {}
            while line:
                if should_ignore_line(line):
                    break
                token = ConllToken(line, phantom_tokens)

                if token.discard_sentence:
                    return False

                if not token.is_phantom:
                    self.tokens.append(token)
                line = file_handle.readline()
            return True

        while (not read_next_sentence()):
            skip_sentence()
        if self.tokens:
            assert(len(self.tokens) == self.tokens[-1].id)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

    def words(self):
        return [token.form for token in self.tokens]

    def get_tree_relation(self, left, right):
        def traverse_ancestors(node_idx):
            visited = set()
            next_node = node_idx
            while 0 not in visited:
                assert(next_node not in visited)
                visited.add(next_node)
                node = self.tokens[next_node-1]
                next_node = node.head_idx
            return visited

        assert(left < right)
        if self.tokens[right].head_idx == left:
            return TreeRelation.LEFT_HEAD
        if self.tokens[left].head_idx == right:
            return TreeRelation.RIGHT_HEAD
        if left in traverse_ancestors(right):
            return TreeRelation.LEFT_ANCESTOR
        if right in traverse_ancestors(left):
            return TreeRelation.RIGHT_ANCESTOR
        return TreeRelation.DISCONNECTED

class ConllDocument():
    def __init__(self, path, word2idx):
        self.word2idx = word2idx
        self.ids = None

        assert os.path.exists(path)
        self.file = open(path, 'r', encoding="utf8")

    def sentences(self):
        sentence = ConllSentence(self.file)
        while len(sentence) > 0:
            yield sentence
            sentence = ConllSentence(self.file)

    def vectorize(self, sentence):
        ids = torch.LongTensor(len(sentence), 1)
        for i, word in enumerate(sentence.words()):
            if word in self.word2idx:
                ids[i, 0] = self.word2idx[word]
            else:
                ids[i, 0] = self.word2idx['<unk>']
        data = ids[0:-1]
        target = ids[1:]
        return data, target
