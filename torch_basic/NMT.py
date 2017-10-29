import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import FreqDist

import re
import jieba
import math
import time
from collections import Counter
import random

path = '../data/cmn-eng/'

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

# 工具函数
def deal_en_sen(raw):
    raw.strip()
    letters_only = re.sub("[^a-zA-Z]", " ", raw)
    words = letters_only.lower().split()

    return (" ".join(words))

def deal_zh_sen(raw):
    raw.strip()
    letters_only = re.sub("[^\u4e00-\u9fa5]", "", raw)                        
    
    return(letters_only) 

def wordandindex(vocab):
    return {word: i + 1 for i, word in enumerate(vocab)}, {i + 1: word for i, word in enumerate(vocab)}

def sen2index(sen, lang):
    global word2index_en
    global word2index_zh
    if lang == 'en':
        no_eos = [word2index_en[word] for word in sen.split(' ')]
    else:
        no_eos = [word2index_zh[word] for word in list(jieba.cut(sen))]
    no_eos.append(0)
    return no_eos
    
def as_minutes(s):
    pass

def time_since(since, percent):
    pass
    
# 数据预处理
with open(path + 'cmn.txt') as f:
    lines = f.readlines()
    
en_sens = [deal_en_sen(line.split('\t')[0]) for line in lines]
zh_sens = [deal_zh_sen(line.split('\t')[1]) for line in lines]
pairs = [[en, zh] for en, zh in zip (en_sens, zh_sens)] 

en_max_len = max([len(x) for x in en_sens])
zh_max_len = max([len(x) for x in zh_sens])

# 借助 NLTK 函数
en_word_counts = FreqDist(' '.join(en_sens).split(' '))
# zh_word_counts = FreqDist(list(jieba.cut(''.join(zh_sens))))
en_vocab = set(en_word_counts)
# zh_vocab = set(zh_word_counts)
zh_counts = Counter()
for sen in zh_sens:
    for word in list(jieba.cut(sen)):
        zh_counts[word] += 1
        
zh_vocab = set(zh_counts)

MAX_LENGTH = 7
filter_pairs = [pair for pair in pairs if len(pair[0].split(' ')) < MAX_LENGTH and len(list(jieba.cut(pair[1]))) < MAX_LENGTH]

word2index_en, index2word_en = wordandindex(en_vocab)
word2index_en['EOS'] = 0
index2word_en[0] = 'EOS'
word2index_zh, index2word_zh = wordandindex(zh_vocab)
word2index_zh['EOS'] = 0
index2word_zh[0] = 'EOS'
sen_vector = [[sen2index(pair[0], 'en'), sen2index(pair[1], 'zh')] for pair in filter_pairs]

    
# 模型实现
# seq2seq with attention
# np.array([sen_vector[2][1]]).shape
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, is_cuda=False):
        super(EncoderRNN, self).__init__()
        # input_size 实际上是 vocab 的size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.is_cuda = is_cuda
        
        # input (N, W) LongTensor N = mini-batch
        # output (N, W, embedding_dim)
        self.embedded = nn.Embedding(input_size, hidden_size)
        # input (seq_len, batch, input_size)
        # h_0 (num_layers * num_directions, batch, hidden_size)
        # output (seq_len, batch, hidden_size * num_directions)
        # h_n (num_layers * num_directions, batch, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs[0])
        embedded = self.embedded(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        
        return output, hidden
    
    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if self.is_cuda: hidden = hidden.cuda()
        return hidden    
    
# 为了简便 这里实现的是 Attn 中的 general method
class Attn(nn.Module):
    def __init__(self, hidden_size, max_length=MAX_LENGTH, is_cuda=False):
        super(Attn, self).__init__()
        
        self.hidden_size = hidden_size
        self.is_cuda = is_cuda
        
        # general method
        self.attn = nn.Linear(self.hidden_size, hidden_size)
    
    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        
        attn_energies = Variable(torch.zeros(seq_len))
        if self.is_cuda: attn_energies = attn_energies.cuda()
        
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden.squeeze(0).squeeze(0), encoder_outputs[i].squeeze(0))
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        # 返回的 是 attn_weigths 维度 与 encoder_outputs 保持一致
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
        
    
    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = hidden.dot(energy)
        return energy

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, is_cuda=False):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.is_cuda = is_cuda
        
        # outout_size 为 中文 vocab 的 length
        self.embedded = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size)
        # input => (N, *, in_features)
        # output => (N, * , out_features)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.attn = Attn(hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # embedding 的本来输出是 (n, w, embedding_dim)
        # [1, 1, 10]
        embedded = self.embedded(word_input)
        # print("decoder's embedded is {}".format(embedded))
        # [1, 1, 20]
        rnn_input = torch.cat((embedded,last_context), 2)
        # print("decoder's rnn_input is {}".format(rnn_input))
        # [1, 1, 10]
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        # print("decoder's rnn_output is {}".format(rnn_output))
        # [1, 1, 3]
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # print("decoder's attn_weights is {}".format(attn_weights))
        # [1, 1, 3] bmm [1, 3, 10](转置之前 [3, 1, 10]) => [1, 1, 10]
        # print(type(attn_weights))
        # print(type(encoder_outputs.transpose(0, 1)))
        # print(attn_weights.size())
        # print(encoder_outputs.transpose(0,1).size())
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # print("decoder's context is {}".format(context))
        #print("{}".format(self.out(torch.cat((rnn_output, context), 2)).size()))
        output = F.log_softmax(self.out(torch.cat((rnn_output.squeeze(0), context.squeeze(0)), 1)))
        #print("decoder's output is {}".format(output))
        return output, context, hidden, attn_weights
          
# 训练
# 500
hidden_size = 128
n_layers = 1
MAX_LENGTH = 7

# USE_CUDA = False
USE_CUDA = False

encoder = EncoderRNN(len(en_vocab), hidden_size, n_layers, False)
decoder = DecoderRNN(hidden_size, len(zh_vocab), n_layers, False)

if USE_CUDA:
    encoder.cuda()
    decoder.cuda()
    
lr = 1e-3

encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.NLLLoss()
# 训练函数
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, is_cuda=False):
    # 梯度初始化
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # added noto for each word
    
    # input_length = input_variable.size()[1]
    target_length = target_variable.size()[1]
    
    ## encoder
    # [num_layers * direc, batch_s, fea] [1, 1, 500]
    encoder_hidden = encoder.init_hidden()
    # 假设 input 为 [1, 4, 5, 7] [1, 4]
    # [4, 1, 500] [1, 1, 500]
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    ## decoder
    # [1, 1]
    decoder_input = Variable(torch.LongTensor([[0]]))
    # [1, 1, 500]  
    decoder_context = Variable(torch.zeros(1, 1, decoder.hidden_size))
    # [1, 1, 500]
    decoder_hidden = encoder_hidden
    
    if is_cuda:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        
    for i in range(target_length):
        
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        # ???
        # print(target_variable[i].size())
        loss += criterion(decoder_output, target_variable[0][i])
        topv, topi = decoder_output.data.topk(1)
        max_similar_pos = topi[0][0]
        decoder_input = Variable(torch.LongTensor([[max_similar_pos]]))
        # print(decoder_input)
        if USE_CUDA: decoder_input = decoder_input.cuda()
        if max_similar_pos == 0: break
            
    
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length
    
    
# 训练细节定义
n_epochs = 500
print_every = 50
print_loss_total = 0

# 开始训练
for epoch in range(1, n_epochs + 1):
    
    training_pair = random.choice(sen_vector)
    
    input_variable = Variable(torch.LongTensor([training_pair[0]]))
    # int(input_variable.size())
    target_variable = Variable(torch.LongTensor([training_pair[1]]))
    
    if USE_CUDA:
        input_variable = input_variable.cuda()
        target_variable = target_variable.cuda()
    
    loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, False)
    
    print_loss_total += loss
    
    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / epoch
        print('epoch {}\'s avg_loss is {}'.format(epoch, print_loss_avg))
        
        
# 查看结果
def evaluate(sentence, max_length=MAX_LENGTH, is_cuda=False):
    input_sen2index = sen2index(sentence, 'en')
    input_variable = Variable(torch.LongTensor(input_sen2index).view(1, -1))
    input_length = input_variable.size()[1]
    
    if is_cuda:
        input_variable = input_variable.cuda()
    
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    decoder_input = Variable(torch.LongTensor([[0]]))
    decoder_context = Variable(torch.zeros(1, 1, decoder.hidden_size))
                            
    if is_cuda:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        
    decoder_hidden = encoder_hidden   
    # 翻译结果
    decoded_words = []
    # 这块还不是很理解
    decoder_attentions = torch.zeros(max_length, max_length)
    
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention =  decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        # EOS_token
        if ni == 0:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(index2word_zh[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
            
    return ''.join(decoded_words), decoder_attentions[:di+1, :len(encoder_outputs)]    
                                  
print(evaluate('i love you')[0])
    
    
    
    
    
    
    