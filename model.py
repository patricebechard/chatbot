#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Thu Jan 18 14:08:24 2018

Sequence to sequence chatbot

"""

from utils import USE_CUDA
from preprocessing import MAX_LENGTH

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding, n_layers=1, cell_type='LSTM',
                 attn_status=False, teacher_forcing_ratio=0., dropout=0.,
                 learning_rate=0.01, max_length=MAX_LENGTH):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = embedding

        if cell_type == 'LSTM':
            self.hidden_cells = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers)
        elif cell_type == 'GRU':
            self.hidden_cells = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        else:
            raise Exception ("Cell type %s for encoder is unsupported"
                             % cell_type)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.hidden_cells(embedded, hidden)

        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, embedding, n_layers=1, cell_type='LSTM',
                 attn_status=False, teacher_forcing_ratio=0., dropout=0.,
                 learning_rate=0.01, max_length=MAX_LENGTH):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = embedding
        self.n_layers = n_layers

        if cell_type == 'LSTM':
            self.hidden_cells = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers)
        elif cell_type == 'GRU':
            self.hidden_cells = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        else:
            raise Exception ("Cell type %s for encoder is unsupported"
                             % cell_type)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):

        output = self.embedding(input).view(1, 1, -1)

        output, hidden = self.hidden_cells(output, hidden)

        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

def loadModel(vocab_size):

    print("Importing network architecture and parameters ...")

    import json
    parameters = json.load(open('parameters.json'))

    cell_type = parameters['cell_type']
    hidden_size = parameters['hidden_size']
    n_layers = parameters['n_layers']
    attn_status = parameters['attn_status']
    teacher_forcing_ratio = parameters['teacher_forcing_ratio']
    dropout = parameters['dropout']
    learning_rate = parameters['learning_rate']

    embedding = nn.Embedding(vocab_size, hidden_size)

    encoder = EncoderRNN(vocab_size, hidden_size, embedding, n_layers,
                         cell_type=cell_type, attn_status=attn_status,
                         teacher_forcing_ratio=teacher_forcing_ratio,
                         dropout=dropout, learning_rate=learning_rate,
                         max_length=MAX_LENGTH)
    decoder = DecoderRNN(hidden_size, vocab_size, embedding, n_layers,
                         cell_type=cell_type, attn_status=attn_status,
                         teacher_forcing_ratio=teacher_forcing_ratio,
                         dropout=dropout, learning_rate=learning_rate,
                         max_length=MAX_LENGTH)

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    return encoder, decoder
