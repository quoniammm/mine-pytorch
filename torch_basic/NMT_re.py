import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from nltk import FreqDist
from collections import Counter

import numpy as np
import pandas as pd

import time
import math
import re
import jieba
import random

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, is_cuda=False):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.is_cuda = is_cuda
        
        # input (n, w)
        self.embedded = nn.Embedding(input_size, hidden_size)
        # input (seq, batch_s, fea)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)
        
     
    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs[0])
        # output (n, w, e_dim) view (w, n, e_dim)
        embedded = self.embedded(word_inputs).view(seq_len, 1, -1)
        # output (seq, batch_s, hidden * direc) hidden (n_layer * direc, batch_s, fea)
        output, hidden = self.gru(embedded, hidden)
        
        return output, hidden
    
    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.is_cuda: hidden = hidden.cuda()
        return hidden