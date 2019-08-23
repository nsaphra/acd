import torch.nn as nn
from collections import namedtuple, defaultdict
import torch
import pandas
import numpy

class AnalysisHook:
    def __init__(self, key, analyzer):
        Stat = namedtuple('Stat', ['name', 'func'])

        self.stat_functions = []

        def fano_factor(x):
            mean = x.abs().mean(dim=-1, keepdim=True)
            square_mean = x.pow(2).mean(dim=-1, keepdim=True)

            return mean/square_mean - mean

        if analyzer.l1:
            self.stat_functions.append(Stat('l1', lambda x: x.norm(1, dim=-1, keepdim=True)))
        if analyzer.l2:
            self.stat_functions.append(Stat('l2', lambda x: x.norm(2, dim=-1, keepdim=True)))
        if analyzer.mean:
            self.stat_functions.append(Stat('mean', lambda x: x.mean(dim=-1, keepdim=True)))
        if analyzer.magnitude:
            self.stat_functions.append(Stat('magnitude', lambda x: x.abs().max(dim=-1, keepdim=True)[0]))
        if analyzer.range:
            self.stat_functions.append(Stat('range', lambda x: x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0]))
        if analyzer.median:
            self.stat_functions.append(Stat('median', lambda x: x.median(dim=-1, keepdim=True)[0]))
        if analyzer.variance:
            self.stat_functions.append(Stat('variance', lambda x: x.var(dim=-1, keepdim=True)))
        if analyzer.fano:
            self.stat_functions.append(Stat('fano', fano_factor))
        if analyzer.concentration:
            self.stat_functions.append(Stat('concentration', lambda x: x.norm(1, dim=-1, keepdim=True)/x.norm(2, dim=-1, keepdim=True)))

        self.running_stats = defaultdict(list)

        self.key = key
        self.analyzer = analyzer

    def dummy_hook(self, layer, grad_input, grad_output):
        print(layer)
        for key, parameter in layer.named_parameters():
            print(key, parameter.size())

        print("output")
        print([x.size() if x is not None else None for x in grad_output])
        print("input")
        print([x.size() if x is not None else None for x in grad_input])

    def hook(self, module, input, output, backward_pass=False):
        if backward_pass:
            direction = 'backward'
        else:
            direction = 'forward'

        start_indices, stop_indices = self.analyzer.start_indices, self.analyzer.stop_indices
        if start_indices == None:
            return

        for i in range(len(start_indices)):
            stop_idx = stop_indices[i]
            start_idx = start_indices[i]

            for idx, vectors in enumerate(self.process_gradient(output)):
                self.store_stats('{}_{}_out_{}'.format(self.key, direction, idx), vectors[start_idx:stop_idx+1])
            for idx, vectors in enumerate(self.process_gradient(input)):
                self.store_stats('{}_{}_in_{}'.format(self.key, direction, idx), vectors[start_idx:stop_idx+1])

    def forward_hook(self, module, input, output):
        return self.hook(module, input, output, backward_pass=False)

    def backward_hook(self, module, input, output):
        return self.hook(module, input, output, backward_pass=True)

    def store_stats(self, direction_key, vectors):
        for stat_function in self.stat_functions:
            sequence_stats = stat_function.func(vectors)
            for i,stat in enumerate(sequence_stats):
                key = '{}_{}_{}'.format(direction_key, stat_function.name, i)
                self.running_stats[key].append(float(stat.detach().cpu().numpy()))

    def process_gradient(self, gradients):
        if gradients is None or len(gradients) == 0:
            return
        if type(gradients) is torch.Tensor and gradients.dtype not in (torch.double, torch.float):
            return

        def matches_sequence_length(d1, d2):
            return d1 * d2 == self.analyzer.sequence.size(0)

        def should_collapse_tuple(length, first_elt):
            if matches_sequence_length(length, first_elt.size(0)):
                return True
            return first_elt.size(0) == 1 and matches_sequence_length(length, first_elt.size(1))

        if type(gradients) is tuple:
            if type(gradients[0]) is torch.cuda.FloatTensor and should_collapse_tuple(len(gradients), gradients[0]):
                gradients = torch.stack(gradients, dim=0)
            else:
                for gradient in gradients:
                    for processed_gradient in self.process_gradient(gradient):
                        yield processed_gradient
                return

        if gradients.dim() == 3:
            if matches_sequence_length(gradients.size(0), gradients.size(1)):
                # gradients: sequence_length x batch_size x hidden_size
                gradients = gradients.view(self.analyzer.sequence.size(0), -1)

        if gradients.size(0) == self.analyzer.sequence.size(0):
            yield gradients
        # activations: (sequence_length * batch_size) x hidden_size

    def register_backward_hook(self, module):
        self.backward_handle = module.register_backward_hook(self.backward_hook)

    def register_forward_hook(self, module):
        self.forward_handle = module.register_forward_hook(self.forward_hook)

    def remove_backward_hook(self):
        if self.backward_handle is not None:
            self.backward_handle.remove()

    def remove_forward_hook(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()

    def clear_stats(self):
        self.running_stats = defaultdict(list)

    def stat_key(self, layer, stat):
        return '_'.join([self.key, str(layer), stat.name])

    def serialize_stats(self):
        return pandas.DataFrame(self.running_stats)

class GradientAnalyzer:
    """
    exists so we can add the AnalysisHook onto the layers, and clear them
    out at different points
    hooks are the analysis hook handles
    """
    def __init__(self, model,
                 l1=False, l2=False, mean=False, magnitude=False, range=False, median=False, variance=False, fano=False, concentration=False):
        self.model = model
        self.hooks = {}
        self.external_stats = defaultdict(list)

        self.l1 = l1
        self.l2 = l2
        self.mean = mean
        self.magnitude = magnitude
        self.range = range
        self.median = median
        self.variance = variance
        self.fano = fano
        self.concentration = concentration

    @staticmethod
    def module_output_size(module):
        # return the size of the final parameters in the module,
        # or 0 if there are no parameters
        output_size = 0
        for key, parameter in module.named_parameters():
            if key.find('weight') < 0:
                continue
            output_size = parameter.size(-1)
        return output_size

    def set_word_sequence(self, sequence, index_pairs):
        self.sequence = sequence.view(-1)
        self.start_indices = index_pairs[0]
        self.stop_indices = index_pairs[1]

    def add_hooks_recursively(self, parent_module: nn.Module, prefix=''):
        # add hooks to the modules in a network recursively
        for module_key, module in parent_module.named_children():
            module_key = prefix + module_key
            output_size = self.module_output_size(module)
            if output_size == 0:
                continue
            self.hooks[module_key] = AnalysisHook(module_key, self)

            self.hooks[module_key].register_forward_hook(module)
            self.hooks[module_key].register_backward_hook(module)
            self.add_hooks_recursively(module, prefix=module_key)

    def add_hooks_to_model(self):
        self.add_hooks_recursively(self.model)

    def remove_hooks(self):
        for key, hook in self.hooks.items():
            hook.remove_forward_hook()
            hook.remove_backward_hook()
            del self.hooks[key]

    def store_external_stats(self, key, vectors):
        if self.start_indices == None:
            return

        for i in range(len(self.start_indices)):
            stop_idx = self.stop_indices[i]
            start_idx = self.start_indices[i]

            for i in range(len(self.start_indices)):
                stop_idx = self.stop_indices[i]
                start_idx = self.start_indices[i]

                for offset in range(stop_idx - start_idx + 1):
                    step = start_idx + offset
                    self.external_stats['{}_{}'.format(key, offset)].append(float(vectors[step].detach().cpu().numpy()))

    def clear_external_stats(self):
        self.external_stats = defaultdict(list)

    def serialize_external_stats(self):
        return pandas.DataFrame(self.external_stats)

    def compute_and_clear(self, fname):
        print('printing final statistics to ', fname)
        frame = pandas.DataFrame()

        for key, hook in self.hooks.items():
            current = hook.serialize_stats()
            frame = pandas.concat([frame, current], axis=1)
            hook.clear_stats()

        frame = pandas.concat([frame, self.serialize_external_stats()], axis=1)
        self.clear_external_stats()
        with open(fname, 'w') as file:
            frame.to_csv(file, encoding='utf-8')
