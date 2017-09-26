#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import re
import time
import jieba
import random
import math
import string

#%%
# # 1.数据处理部分
USE_CUDA = False
path = 'data/cmn-eng/'
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

def isChinese(sen):
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    return zhPattern.search(sen)
# 简化句子 便于处理
def normalize_string(s):
    s = re.sub(r"[!！？.()（）""?。“”，,']", r" ", s)
    return s

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS
      
    def index_words(self, sentence):
        sen_list = []
        if isChinese(sentence):
            sen_list = jieba.cut(sentence)
        else:
            sen_list = sentence.split(' ')
            
        for word in sen_list:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_sen(path, lang1, lang2, reverse=False):
    with open(path + '{}-{}.txt'.format(lang1, lang2)) as f:
        lines = f.readlines()
        pairs = []
        for line in lines:
            line = line.strip()
            if reverse:
                line = line.split('\t')
                line.reverse()
                line = "\t".join(line)
                
            pair = [normalize_string(sen) for sen in line.split('\t')]
            pairs.append(pair)
        
        if reverse:
            input_lang = Lang(lang2)
            output_lang = Lang(lang1) 
        else:
            input_lang = Lang(lang1)            
            output_lang = Lang(lang2)   

        print("input_lang is {}".format(input_lang.n_words))
            
    return input_lang, output_lang, pairs

def data_preprocess(path, lang1, lang2, reverse=False):
    print("Read lines......")
    input_lang, output_lang, pairs = read_sen(path, lang1, lang2, reverse)
    print("Trimmed  to {} sentence pairs".format(len(pairs)))
    
    print("Indexing words......")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    
    return input_lang, output_lang, pairs
    
input_lang, output_lang, pairs = data_preprocess(path, 'eng', 'cmn')
for i in range(5):
    print(random.choice(pairs))

#%%
# # 2.pytorch 搭建模型
# ## 2.1.数据部分
def indexes_from_sentence(lang, sen):
    if isChinese(sen):
        sen = jieba.cut('')
    else:
        sen = sen.split(' ')
        
    return [lang.word2index[word] for word in sen]

def variable_from_sentence(lang, sen):
    ixs = indexes_from_sentence(lang, sen)
    ixs.append(EOS_token)
    var = Variable(torch.LongTensor(ixs).view(-1, 1))
    if USE_CUDA: 
        var = var.cuda()
    
    return var
    

def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    output_variable = variable_from_sentence(output_lang, pair[1])

    return (input_variable, output_variable)


#%%
# ## 2.2.模型搭建
# 编码层
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(1, seq_len, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden
# Attn 层
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)

        # else self.method = 'concat':
        #     self.attn = 
        #     self.other = 

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size()[1]

        attn_energies = Variable(torch.zeros(seq_len))
        if USE_CUDA:
            attn_energies.cuda()

        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[0][i])

        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        if self.method == 'general':
            energy = self.attn(encoder_output)
            # 矩阵维度有些不理解
            energy = torch.dot(hidden.view(-1), energy.view(-1))
            return energy
# 改进的解码层
class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=.1):
        super(AttnDecoderRNN, self).__init__()
        # 定义参数
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # 定义层
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
        # 为什么乘 2
        self.out = nn.Linear(hidden_size * 2, output_size)

        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        word_embedded = self.embedding(word_input).view(1, 1, -1)

        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)
        # print("context size is {}".format(context.size()))
        rnn_output = rnn_output.squeeze(1)
        context =  context.squeeze(0)
        # print("context size is {}".format(context.size()))        
        # 这块还有点不理解
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        return output, context, hidden, attn_weights

#%%
# 对模型进行测试
encoder_test = EncoderRNN(10, 10, 2)
decoder_test = AttnDecoderRNN('general', 10, 10, 2)

print(encoder_test)
print(decoder_test)

encoder_hidden = encoder_test.init_hidden()
word_input = Variable(torch.LongTensor([1, 9, 3, 4]))

if USE_CUDA:
    encoder_test.cuda()
    word_input.cuda()

encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

word_inputs = Variable(torch.LongTensor([1, 2, 6, 6, 8]))
# 不是很理解
decoder_attns = torch.zeros(1, 5, 4)
decoder_hidden = encoder_hidden 
decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))

if USE_CUDA:
    decoder_test.cuda()
    word_inputs = word_inputs.cuda()
    decoder_context = decoder_context.cuda()

for i in range(5):
    decoder_output, decoder_context, deocder_hidden, decoder_attn = decoder_test(word_inputs[i], decoder_context, decoder_hidden, encoder_outputs)
    decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data
    print(decoder_attns)

     
#%%
# 训练
teacher_forcing_ratio = .5
clip = .5

attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = .5

encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

if USE_CUDA:
    encdoer.cuda()
    decoder.cuda()

learning_rate = .0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

#评判标准
criterion = nn.NLLLoss()

n_epochs = 50000
plot_every = 200
print_every = 1000

start = time.time()
plot_losses = []
print_loss_total = 0
plot_loss_total = 0

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,  criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
  
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_input.cuda()

    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break
        
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)   
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]/target_length


# Begin!
for epoch in range(1, n_epochs + 1):
    
    # Get training data for this cycle
    training_pair = variables_from_pair(random.choice(pairs))
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    # Run the train function
    loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0: continue

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0