import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

import random

from utils import device

import sys

class Seq2Seq(nn.Module):
    
    # since the model is used as a chatbot, the input size is the vocabulary size 
    # and the output size is the same.
    # We only use LSTMs

    def __init__(self, input_size, embedding_size=256, hidden_size=256, 
                 num_layers=3, bidirectional=False, attention=False, 
                 teacher_forcing_prob=1, max_length=30):
        super(Seq2Seq, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.teacher_forcing_prob = teacher_forcing_prob
        self.max_length = max_length

        self.encoder = Encoder(input_size=input_size, embedding_size=embedding_size,
                               hidden_size=hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional)
        self.decoder = Decoder(output_size=input_size, embedding_size=embedding_size, 
                               hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, inputs, targets):

        # feeding data to encoder
        encoder_out = self.encoder(inputs)

        # the hidden and cell states are shared between the encoder and decoder
        self.decoder.hidden = self.encoder.hidden
        self.decoder.cell = self.encoder.cell
        

        if random.random() < self.teacher_forcing_prob:
            # we have teacher forcing, we feed the offset targets as inputs to the decoder
            teacher_forcing = True
            
            decoder_in = torch.zeros_like(targets).to(device)
            decoder_out = torch.zeros(targets.shape[0], targets.shape[1], self.input_size).to(device)
            predicted_words = torch.ones_like(targets).to(device)
            decoder_in[1:] = targets[:-1]

        else:
            # no teacher forcing, we feed the predicted words as input to next timestep
            teacher_forcing = False
            
            decoder_in = torch.zeros(self.max_length, targets.shape[1], dtype=torch.long).to(device)
            decoder_out = torch.zeros(self.max_length, targets.shape[1], self.input_size).to(device)
            predicted_words = torch.ones_like(decoder_in)        
        
        # feeding data to decoder
        for i in range(len(decoder_in)):
            
            decoder_out[i] = self.decoder(decoder_in[i].unsqueeze(0)).squeeze(0)
            
            predicted_words[i] = torch.argmax(decoder_out[i], dim=-1)
            
            if not teacher_forcing and i > self.max_length:
                # input of next timestep is predicted words of this timestep
                decoder_in[i+1] = predicted_words[i]
            
            # if all generated elements are <EOS> (id = 1, -1 -> all() is False)
            if not (predicted_words[i] - 1).byte().all():                
                # trim rest of sequence and get out of loop
                predicted_words = predicted_words[:i+1]
                decoder_out = decoder_out[:i+1]
                break
                
        return decoder_out, predicted_words

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size=256, hidden_size=256, 
                 num_layers=3, bidirectional=False):
        """
        Parameters
        ----------
        input_size : int
            size of vocabulary for input sequence
        embedding_size : int
            size of embedding
        hidden_size : int
            size of hidden state and cell state (if applicable)
        num_layers : int
            number of hidden layers of encoder
        bidirectional : boolean
            whether or not the encoder is a bidirectional rnn or not
        """
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_bidirectional = bidirectional

        self.hidden0 = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
        self.cell0 = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
        
        # initialize with Glorot
        nn.init.xavier_uniform_(self.hidden0)
        nn.init.xavier_uniform_(self.cell0)

        self.embedding = nn.Embedding(num_embeddings=input_size,
                                      embedding_dim=embedding_size)

        self.encoder_rnn = nn.LSTM(input_size=embedding_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional)

    def forward(self, x):
        
        # hidden at t=0 is a learned bias parameter
        self._init_hidden(batch_size=x.shape[-1])

        x = self.embedding(x)
        x, (self.hidden, self.cell) = self.encoder_rnn(x, (self.hidden, self.cell))

        return x

    def _init_hidden(self, batch_size):

        self.hidden = self.hidden0.clone().repeat(1, batch_size, 1).to(device)
        self.cell = self.cell0.clone().repeat(1, batch_size, 1).to(device)

class Decoder(nn.Module):

    def __init__(self, output_size, embedding_size=256, hidden_size=256,
                 num_layers=3):
        """
        Parameters
        ----------
        embedding_size : int
            size of embedding
        hidden_size : int
            size of hidden state and cell state (if applicable)
        num_layers : int
            number of hidden layers of decoder
        """
        super(Decoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size)
        
        self.decoder_rnn = nn.LSTM(input_size=embedding_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers)

        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):

        x = self.embedding(x)     
        x, (self.hidden, self.cell) = self.decoder_rnn(x, (self.hidden, self.cell))
        x = self.fc(x)

        return x
        
if __name__ == "__main__":

	model = Seq2Seq(input_size, )


