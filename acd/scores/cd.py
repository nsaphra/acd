import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from scipy.special import expit as sigmoid_

# propagate a three-part
def propagate_three(a, b, c, activation):
    activation(a)
    a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
    return a_contrib, b_contrib, activation(c)


# propagate tanh nonlinearity
def propagate_tanh_two(a, b):
    tanh = torch.nn.Tanh()
    return 0.5 * (tanh(a) + (tanh(a + b) - tanh(b))), 0.5 * (tanh(b) + (tanh(a + b) - tanh(a)))


# propagate convolutional or linear layer
def propagate_conv_linear(relevant, irrelevant, module, device='cuda'):
    bias = module(torch.zeros(irrelevant.size()).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)


# propagate ReLu nonlinearity
def propagate_relu(relevant, irrelevant, activation, device='cuda'):
    swap_inplace = False
    try:  # handles inplace
        if activation.inplace:
            swap_inplace = True
            activation.inplace = False
    except:
        pass
    zeros = torch.zeros(relevant.size()).to(device)
    rel_score = activation(relevant)
    irrel_score = activation(relevant + irrelevant) - activation(relevant)
    if swap_inplace:
        activation.inplace = True
    return rel_score, irrel_score


# propagate maxpooling operation
def propagate_pooling(relevant, irrelevant, pooler, model_type='mnist'):
    if model_type == 'mnist':
        unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        avg_pooler = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        window_size = 4
    elif model_type == 'vgg':
        unpool = torch.nn.MaxUnpool2d(kernel_size=pooler.kernel_size, stride=pooler.stride)
        avg_pooler = torch.nn.AvgPool2d(kernel_size=(pooler.kernel_size, pooler.kernel_size),
                                        stride=(pooler.stride, pooler.stride), count_include_pad=False)
        window_size = 4

    # get both indices
    p = deepcopy(pooler)
    p.return_indices = True
    both, both_ind = p(relevant + irrelevant)
    ones_out = torch.ones_like(both)
    size1 = relevant.size()
    mask_both = unpool(ones_out, both_ind, output_size=size1)

    # relevant
    rel = mask_both * relevant
    rel = avg_pooler(rel) * window_size

    # irrelevant
    irrel = mask_both * irrelevant
    irrel = avg_pooler(irrel) * window_size
    return rel, irrel


# propagate dropout operation
def propagate_dropout(relevant, irrelevant, dropout):
    return dropout(relevant), dropout(irrelevant)


# get contextual decomposition scores for blob
def cd(blob, im_torch, model, model_type='mnist', device='cuda'):
    # set up model
    model.eval()
    im_torch = im_torch.to(device)

    # set up blobs
    blob = torch.FloatTensor(blob).to(device)
    relevant = blob * im_torch
    irrelevant = (1 - blob) * im_torch

    if model_type == 'mnist':
        scores = []
        mods = list(model.modules())[1:]
        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
        relevant, irrelevant = propagate_pooling(relevant, irrelevant,
                                                 lambda x: F.max_pool2d(x, 2, return_indices=True), model_type='mnist')
        relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)

        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[1])
        relevant, irrelevant = propagate_pooling(relevant, irrelevant,
                                                 lambda x: F.max_pool2d(x, 2, return_indices=True), model_type='mnist')
        relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)

        relevant = relevant.view(-1, 320)
        irrelevant = irrelevant.view(-1, 320)

        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[3])
        relevant, irrelevant = propagate_relu(relevant, irrelevant, F.relu)

        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[4])

    else:
        mods = list(model.modules())
        for i, mod in enumerate(mods):
            t = str(type(mod))
            if 'Conv2d' in t or 'Linear' in t:
                if 'Linear' in t:
                    relevant = relevant.view(relevant.size(0), -1)
                    irrelevant = irrelevant.view(irrelevant.size(0), -1)
                relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mod)
            elif 'ReLU' in t:
                relevant, irrelevant = propagate_relu(relevant, irrelevant, mod)
            elif 'MaxPool2d' in t:
                relevant, irrelevant = propagate_pooling(relevant, irrelevant, mod, model_type=model_type)
            elif 'Dropout' in t:
                relevant, irrelevant = propagate_dropout(relevant, irrelevant, mod)
    return relevant, irrelevant

def through_lstm(self, layer, relevant_input, irrelevant_input, cell_state=None):
    sigmoid = torch.nn.Sigmoid()
    weights = layer.state_dict()

    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = torch.chunk(weights['weight_ih_l0'], 4, 0)
    W_hi, W_hf, W_hg, W_ho = torch.chunk(weights['weight_hh_l0'], 4, 0)
    b_i, b_f, b_g, b_o = torch.chunk(weights['bias_ih_l0'] + weights['bias_hh_l0'], 4, 0)
    T = word_vecs.size(0)
    hidden_dim = word_vecs.size(-1)
    relevant = torch.cuda.FloatTensor(T, hidden_dim).fill_(0)
    irrelevant = torch.cuda.FloatTensor(T, hidden_dim).fill_(0)
    relevant_h = torch.cuda.FloatTensor(T, hidden_dim).fill_(0)
    irrelevant_h = torch.cuda.FloatTensor(T, hidden_dim).fill_(0)

    if cell_state is None:
        prev_irrel_h = torch.cuda.FloatTensor(hidden_dim).fill_(0)
        start_state = torch.cuda.FloatTensor(hidden_dim).fill_(0)
    else:
        prev_irrel_h = cell_state[0][0,0]
        start_state = cell_state[1][0,0]
    prev_rel_h = torch.cuda.FloatTensor(hidden_dim).fill_(0)

    for i in range(T):
        if i > 0:
            prev_rel_h = relevant_h[i - 1]
            prev_irrel_h = irrelevant_h[i - 1]

        rel_i = torch.mv(W_hi, prev_rel_h) + torch.mv(W_ii, relevant_input)
        rel_g = torch.mv(W_hg, prev_rel_h) + torch.mv(W_ig, relevant_input)
        rel_f = torch.mv(W_hf, prev_rel_h) + torch.mv(W_if, relevant_input)
        rel_o = torch.mv(W_ho, prev_rel_h) + torch.mv(W_io, relevant_input)
        irrel_i = torch.mv(W_hi, prev_irrel_h) + torch.mv(W_ii, irrelevant_input)
        irrel_g = torch.mv(W_hg, prev_irrel_h) + torch.mv(W_ig, irrelevant_input)
        irrel_f = torch.mv(W_hf, prev_irrel_h) + torch.mv(W_if, irrelevant_input)
        irrel_o = torch.mv(W_ho, prev_irrel_h) + torch.mv(W_io, irrelevant_input)

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, b_i, sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, b_g, torch.nn.Tanh())

        relevant[i] = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
        irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

        #TODO try considering bias to relevant depending on where we are in the sequence
        # the current system considers all independent contributions of the upper layers to be irrelevant
        # if i >= start and i < stop:
        #     relevant[i] += bias_contrib_i * bias_contrib_g
        # else:
        irrelevant[i] += bias_contrib_i * bias_contrib_g

        rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, b_f, sigmoid)
        if i > 0:
            relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
            irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[i - 1] + irrel_contrib_f * \
                                                                                                      relevant[i - 1]
        else:
            irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * start_state

        o = sigmoid(torch.mv(W_io, word_vecs[i]) + torch.mv(W_ho, prev_rel_h + prev_irrel_h) + b_o)
        rel_contrib_o, irrel_contrib_o, bias_contrib_o = propagate_three(rel_o, irrel_o, b_o, sigmoid)
        new_rel_h, new_irrel_h = propagate_tanh_two(relevant[i], irrelevant[i])
        # relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
        # irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
        relevant_h[i] = o * new_rel_h
        irrelevant_h[i] = o * new_irrel_h

# batch of [start, stop) with unigrams working
def cd_lstm(layer, word_vecs, start, stop, cell_state=None):
    sigmoid = torch.nn.Sigmoid()
    weights = layer.state_dict()

    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = torch.chunk(weights['weight_ih_l0'].t(), 4, 1)
    W_hi, W_hf, W_hg, W_ho = torch.chunk(weights['weight_hh_l0'].t(), 4, 1)
    b_i, b_f, b_g, b_o = torch.chunk(weights['bias_ih_l0'] + weights['bias_hh_l0'], 4, 0)
    T = word_vecs.size(0)
    batch_size = word_vecs.size(1)
    hidden_dim = word_vecs.size(-1)

    relevant = torch.cuda.FloatTensor(T, batch_size, hidden_dim).fill_(0)
    irrelevant = torch.cuda.FloatTensor(T, batch_size, hidden_dim).fill_(0)
    relevant_h = torch.cuda.FloatTensor(T, batch_size, hidden_dim).fill_(0)
    irrelevant_h = torch.cuda.FloatTensor(T, batch_size, hidden_dim).fill_(0)

    if cell_state is None:
        start_state = torch.cuda.FloatTensor(batch_size, hidden_dim).fill_(0)
        prev_irrel_h = torch.cuda.FloatTensor(batch_size, hidden_dim).fill_(0)
    else:
        prev_irrel_h = cell_state[0][0].clone()
        start_state = cell_state[1][0]
    prev_rel_h = torch.cuda.FloatTensor(batch_size, hidden_dim).fill_(0)

    for i in range(T):
        if i > 0:
            prev_rel_h = relevant_h[i - 1]
            prev_irrel_h = irrelevant_h[i - 1]

        rel_i = torch.mm(prev_rel_h, W_hi)
        rel_g = torch.mm(prev_rel_h, W_hg)
        rel_f = torch.mm(prev_rel_h, W_hf)
        rel_o = torch.mm(prev_rel_h, W_ho)
        irrel_i = torch.mm(prev_irrel_h, W_hi)
        irrel_g = torch.mm(prev_irrel_h, W_hg)
        irrel_f = torch.mm(prev_irrel_h, W_hf)
        irrel_o = torch.mm(prev_irrel_h, W_ho)

        if i >= start and i <= stop:
            rel_i = rel_i + torch.mm(word_vecs[i], W_ii)
            rel_g = rel_g + torch.mm(word_vecs[i], W_ig)
            rel_f = rel_f + torch.mm(word_vecs[i], W_if)
            rel_o = rel_o + torch.mm(word_vecs[i], W_io)
        else:
            irrel_i = irrel_i + torch.mm(word_vecs[i], W_ii)
            irrel_g = irrel_g + torch.mm(word_vecs[i], W_ig)
            irrel_f = irrel_f + torch.mm(word_vecs[i], W_if)
            irrel_o = irrel_o + torch.mm(word_vecs[i], W_io)


        rel_contrib_i, irrel_contrib_i, bias_contrib_i = propagate_three(rel_i, irrel_i, b_i, sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = propagate_three(rel_g, irrel_g, b_g, torch.nn.Tanh())

        relevant[i] = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
        irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

        if i >= start and i < stop:
            relevant[i] += bias_contrib_i * bias_contrib_g
        else:
            irrelevant[i] += bias_contrib_i * bias_contrib_g

        rel_contrib_f, irrel_contrib_f, bias_contrib_f = propagate_three(rel_f, irrel_f, b_f, sigmoid)

        if i > 0:
            relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
            irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[i - 1] + irrel_contrib_f * relevant[i - 1]
        else:
            irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * start_state

        o = sigmoid(torch.mm(word_vecs[i], W_io) + torch.mm(prev_rel_h + prev_irrel_h, W_ho) + b_o)
        # rel_contrib_o, irrel_contrib_o, bias_contrib_o = propagate_three(rel_o, irrel_o, b_o, sigmoid)
        new_rel_h, new_irrel_h = propagate_tanh_two(relevant[i], irrelevant[i])
        relevant_h[i] = o * new_rel_h
        irrelevant_h[i] = o * new_irrel_h

    return relevant_h, irrelevant_h

# this implementation of cd is very long so that we can view CD at intermediate layers
# in reality, this should be a loop which uses the above functions
def cd_track_vgg(blob, im_torch, model, model_type='vgg'):
    # set up model
    model.eval()

    # set up blobs
    blob = torch.cuda.FloatTensor(blob)
    relevant = blob * im_torch
    irrelevant = (1 - blob) * im_torch

    mods = list(model.modules())[2:]
    scores = []
    #         (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (1): ReLU(inplace)
    #         (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (3): ReLU(inplace)
    #         (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[0])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[1])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[2])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[3])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[4], model_type=model_type)

    #         (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (6): ReLU(inplace)
    #         (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (8): ReLU(inplace)
    #         (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[5])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[6])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[7])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[8])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[9], model_type=model_type)

    #         (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (11): ReLU(inplace)
    #         (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (13): ReLU(inplace)
    #         (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (15): ReLU(inplace)
    #         (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[10])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[11])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[12])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[13])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[14])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[15])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[16], model_type=model_type)
    #         scores.append((relevant.clone(), irrelevant.clone()))
    #         (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (18): ReLU(inplace)
    #         (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (20): ReLU(inplace)
    #         (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (22): ReLU(inplace)
    #         (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[17])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[18])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[19])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[20])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[21])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[22])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[23], model_type=model_type)
    #         scores.append((relevant.clone(), irrelevant.clone()))
    #         (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (25): ReLU(inplace)
    #         (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (27): ReLU(inplace)
    #         (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (29): ReLU(inplace)
    #         (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[24])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[25])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[26])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[27])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[28])
    scores.append((relevant.clone(), irrelevant.clone()))
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[29])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[30], model_type=model_type)
    #         scores.append((relevant.clone(), irrelevant.clone()))

    relevant = relevant.view(relevant.size(0), -1)
    irrelevant = irrelevant.view(irrelevant.size(0), -1)

    #       (classifier): Sequential(
    #         (0): Linear(in_features=25088, out_features=4096)
    #         (1): ReLU(inplace)
    #         (2): Dropout(p=0.5)
    #         (3): Linear(in_features=4096, out_features=4096)
    #         (4): ReLU(inplace)
    #         (5): Dropout(p=0.5)
    #         (6): Linear(in_features=4096, out_features=1000)
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[32])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[33])
    relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[34])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[35])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[36])
    relevant, irrelevant = propagate_dropout(relevant, irrelevant, mods[37])
    relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[38])

    return relevant, irrelevant, scores
