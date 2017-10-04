import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

d = pd.read_json('imdb_final.json')
d['rating'] = d['rating'] - 1
d = d[['tokens','rating']]
X = d.tokens
y = d.rating
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size = 0.3, random_state= 42)
y_train.shape

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)

class AttentionWordRNN(nn.Module):
    
    
    def __init__(self, batch_size, num_tokens, embed_size, word_gru_hidden, bidirectional= True):        
        
        super(AttentionWordRNN, self).__init__()
        
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        
        self.lookup = nn.Embedding(num_tokens, embed_size)
        if bidirectional == True:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2* word_gru_hidden,2*word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2* word_gru_hidden,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2*word_gru_hidden, 1))
        else:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= False)
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
            
        self.softmax_word = nn.Softmax()
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)

        
    # 权重??? state_word???  
    def forward(self, embed, state_word):
        # embeddings
        embedded = self.lookup(embed)
        # word level gru
        output_word, state_word = self.word_gru(embedded, state_word)
#         print output_word.size()
        word_squish = batch_matmul_bias(output_word, self.weight_W_word,self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1,0))
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1,0))        
        return word_attn_vectors, state_word, word_attn_norm
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))

class AttentionSentRNN(nn.Module):
    def __init__(self, batch_size, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional= True):        
        
        super(AttentionSentRNN, self).__init__()
        
        self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        
        
        if bidirectional == True:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional= True)        
            self.weight_W_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden ,2* sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden,1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden, 1))
            self.final_linear = nn.Linear(2* sent_gru_hidden, n_classes)
        else:
            self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional= True)        
            self.weight_W_sent = nn.Parameter(torch.Tensor(sent_gru_hidden ,sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(sent_gru_hidden,1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            self.final_linear = nn.Linear(sent_gru_hidden, n_classes)
        self.softmax_sent = nn.Softmax()
        self.final_softmax = nn.Softmax()
        self.weight_W_sent.data.uniform_(-0.1, 0.1)
        self.weight_proj_sent.data.uniform_(-0.1,0.1)
        
        
    def forward(self, word_attention_vectors, state_sent):
        output_sent, state_sent = self.sent_gru(word_attention_vectors, state_sent)        
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent,self.bias_sent, nonlinearity='tanh')
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_norm = self.softmax_sent(sent_attn.transpose(1,0))
        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1,0))        
        # final classifier
        final_map = self.final_linear(sent_attn_vectors.squeeze(0))
        return F.log_softmax(final_map), state_sent, sent_attn_norm
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden))   

def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
        
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
def pad_batch(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(x) for x in mini_batch]))
    max_token_len = int(np.max([len(val) for sublist in mini_batch for val in sublist]))
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i,j,k] = mini_batch[i][j][k]
                except IndexError:
                    pass
    # sen_len * batch * word_vec
    return Variable(torch.from_numpy(main_matrix).transpose(0,1))
    
    
def gen_minibatch(tokens, labels, mini_batch_size, shuffle=True):
    for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle):
        token = pad_batch(token)
        yield token.cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda()  

def get_predictions(val_tokens, word_attn_model, sent_attn_model):
    max_sents, batch_size, max_tokens = val_tokens.size()
    state_word = word_attn_model.init_hidden().cuda()
    state_sent = sent_attn_model.init_hidden().cuda()  
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(val_tokens[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)            
    y_pred, state_sent, _ = sent_attn_model(s, state_sent)    
    return y_pred
        
def test_accuracy_mini_batch(tokens, labels, mini_batch_size, word_attn, sent_attn):
    p = []
    l = []
    g = gen_minibatch(tokens, labels, mini_batch_size)
    for token, label in g:
        y_pred = get_predictions(token.cuda(), word_attn, sent_attn)
        _, y_pred = torch.max(y_pred, 1)
        p.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
        l.append(np.ndarray.flatten(label.data.cpu().numpy()))
    #??? 不是很明白  
    p = [item for sublist in p for item in sublist]
    l = [item for sublist in l for item in sublist]
    p = np.array(p)
    l = np.array(l)
    num_correct = sum(p == l)
    return float(num_correct)/ len(p)
    
# def check_val_loss():
#     pass

def train_data(mini_batch, targets, word_attn_model, sent_attn_model, word_optimizer, sent_optimizer, criterion):
    state_word = word_attn_model.init_hidden().cuda()
    state_sent = sent_attn_model.init_hidden().cuda()
    max_sents, batch_size, max_tokens = mini_batch.size()
    word_optimizer.zero_grad()
    sent_optimizer.zero_grad()
    s = None
    for i in xrange(max_sents):
        _s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)            
    y_pred, state_sent, _ = sent_attn_model(s, state_sent)
    loss = criterion(y_pred.cuda(), targets) 
    loss.backward()
    
    word_optimizer.step()
    sent_optimizer.step()
    
    return loss.data[0]

word_attn = AttentionWordRNN(
    batch_size=8, 
    num_tokens=8113, 
    embed_size=300,                    
    word_gru_hidden=100, 
    bidirectional= True
)

sent_attn = AttentionSentRNN(
    batch_size=8, 
    sent_gru_hidden=100, 
    word_gru_hidden=100,             
    n_classes=10, 
    bidirectional= True
)

word_attn.cuda()
sent_attn.cuda()
