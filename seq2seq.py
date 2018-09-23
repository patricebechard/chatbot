import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

class Seq2Seq(nn.Module):
    """ 
    Sequence to Sequence model wrapper 
    In the case of a chatbot, the source vocab size and the target vocab size are the same.
    """
    
    def __init__(self, vocab_size, embedding_size=256, hidden_size=256,
                 attention_size=256, num_layers=1, max_seq_len=30):
        super(Seq2Seq, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        self.encoder = Encoder(input_size=vocab_size, 
                               embedding_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers)
        
        self.decoder = Decoder(output_size=vocab_size,
                               embedding_size=embedding_size,
                               hidden_size=hidden_size,
                               attention_size=attention_size,
                               num_layers=num_layers)
        
class Encoder(nn.Module):
    
    def __init__(self, input_size, embedding_size=256, hidden_size=256,
                 num_layers=1):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # initialize to 0 or randn?
        self.hidden0 = nn.Parameter(torch.randn(2*num_layers, 1, hidden_size//2))
        self.cell0 = nn.Parameter(torch.randn(2*num_layers, 1, hidden_size//2))
        
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        self.encoder_rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size//2, 
                                   num_layers=num_layers, bidirectional=True)
        
    def forward(self, x):
        
        hidden = self._init_hidden()
        
        x = self.embedding(x)
        x, hidden = self.encoder_rnn(x, hidden)

        return x, hidden

    def _init_hidden(self):
        # initial hidden state is a learned bias parameter
        hidden = self.hidden0.clone().repeat(1, batch_size, 1)
        cell = self.cell0.clone().repeat(1, batch_size, 1)
        if use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()

        return (hidden, cell)

    
class Decoder(nn.Module):
    
    def __init__(self, output_size, embedding_size=256, hidden_size=256,
                 attention_size=256, num_layers=1):
        """
        We use the original attention mechanism from Bahdanau et al. 2014
        The layers are of this model correspond to the following notation in the original paper.
        
        attn_fc_prev_hid : Wa
        attn_fc_enc_hid : Ua
        attn_fc_context : va
        """
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.num_layers=num_layers
        
        # attention
        self.attn_fc_prev_hid = nn.Linear(in_features=hidden_size, 
                                          out_features=attention_size)
        self.attn_fc_enc_hid = nn.Linear(in_features=hidden_size, 
                                         out_features=attention_size)
        self.attn_fc_context = nn.Linear(in_features=attention_size,
                                         out_features=1)
    
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size)    
        self.decoder_rnn = nn.LSTM(input_size=(embedding_size + hidden_size), 
                                   hidden_size=hidden_size, 
                                   num_layers=num_layers)
        self.clf = nn.Linear(in_features=hidden_size, out_features=output_size)
        
        
    def forward(self, x, encoder_outputs):
        
        x = x.unsqueeze(0)
        
        # attention
        tmp1 = self.attn_fc_prev_hid(self.hidden[0])    # not sure about this for lstm...
        tmp2 = self.attn_fc_enc_hid(encoder_outputs)

        context_weights = self.attn_fc_context(F.tanh(tmp1 + tmp2))
        context_weights = F.softmax(context_weights, 0)
        self.attn_weights = context_weights #for attention plotting purposes
        
        context_weights = context_weights.permute(1, 2, 0) # (batch size x 1 x seq_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # (batch size x seq_len x 2*hidden_dim)

        context_vector = torch.bmm(context_weights, encoder_outputs)
        context_vector = context_vector.permute(1, 0, 2) # (1 x batch_size x 2*hidden_dim)
        
        x = self.embedding(x)
        # concatenate previously predicted word embedding and context vector
        x = torch.cat((x, context_vector), dim=-1)
        x, self.hidden = self.decoder_rnn(x, self.hidden)
        x = self.clf(x)
        
        # for 
        x = F.log_softmax(x, dim=-1)

        return x

    def _init_hidden(self, encoder_hidden_state):
        
        self.hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            self.hidden = self.hidden.cuda()
        for layer in range(self.num_layers):
            self.hidden[layer] = torch.cat((encoder_hidden_state[2*layer],
                                    encoder_hidden_state[2*layer + 1]), dim=-1)
        