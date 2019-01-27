import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch.autograd import Variable

class baselineLSTM(nn.Module):
    def __init__(self, config):
        super(baselineLSTM, self).__init__()
        
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.layers_dim = config['layers']
        self.max_len = config['max_len']

        self.directions = 2 if (config['bidirectional']) else 1

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layers_dim,\
                            batch_first=True, dropout=config['dropout'],\
                            bidirectional=config['bidirectional'])
        self.hidden = None
        self.set_hidden(batch_size=1, zero=True)

        self.out_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=2)

    def set_hidden(self, batch_size, zero=False):
        del self.hidden
        if (zero):
            self.hidden = (torch.zeros(self.layers_dim*self.directions, batch_size, self.hidden_dim).cuda(),\
                           torch.zeros(self.layers_dim*self.directions, batch_size, self.hidden_dim).cuda()) 
        else:
            self.hidden = (torch.randn(self.layers_dim*self.directions, batch_size, self.hidden_dim).cuda(),\
                           torch.randn(self.layers_dim*self.directions, batch_size, self.hidden_dim).cuda())
        
        
    def forward(self, sequence, train=False, init_hidden=False, temp=1.):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        if (init_hidden):
            self.set_hidden(zero=train, batch_size=len(sequence))

        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        out, self.hidden = self.lstm(sequence.cuda(), self.hidden)
        out = self.out_layer(out.contiguous().view(-1, out.size(2)).cuda()) 
        out = torch.div(out, temp)
        out = out.contiguous().view(batch_size, seq_len, self.output_dim).cuda()
        del batch_size, seq_len
        return out, self.softmax(out)

class GRU(nn.Module):
    def __init__(self, config):
        super(GRU, self).__init__()
        
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.layers_dim = config['layers']
        self.max_len = config['max_len']

        self.directions = 2 if (config['bidirectional']) else 1

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.layers_dim,\
                            batch_first=True, dropout=config['dropout'],\
                            bidirectional=config['bidirectional'])
        self.hidden = None
        self.set_hidden(batch_size=1, zero=True)

        self.out_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=2)

    def set_hidden(self, batch_size, zero=False):
        del self.hidden
        if (zero):
            self.hidden = torch.zeros(self.layers_dim*self.directions, batch_size, self.hidden_dim).cuda()
        else:
            self.hidden = torch.randn(self.layers_dim*self.directions, batch_size, self.hidden_dim).cuda()
        
    def forward(self, sequence, train=False, init_hidden=False, temp=1.):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)
        if (init_hidden):
            self.set_hidden(zero=train, batch_size=len(sequence))

        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        out, self.hidden = self.gru(sequence.cuda(), self.hidden)
        out = self.out_layer(out.contiguous().view(-1, out.size(2)).cuda()) 
        out = torch.div(out, temp)
        out = out.contiguous().view(batch_size, seq_len, self.output_dim).cuda()
        return out, self.softmax(out)
