import torch
from torch.autograd import Variable

import pytest
from NMT import EncoderRNN, Attn, DecoderRNN

@pytest.fixture
def encoder():
    encoder = EncoderRNN(10, 10, 2)
    hidden = encoder.init_hidden()
    word_inputs = Variable(torch.LongTensor([[1, 2, 3]]))
    encoder_outputs, encoder_hidden = encoder(word_inputs, hidden)
    return encoder, word_inputs, encoder_outputs, encoder_hidden

# test hidden output 和 init_hidden 的 size 和期望的一样
def testEncoderRNN(encoder):
    # word_inputs
    assert str(encoder[1].size()) == 'torch.Size([1, 3])'
    # encoder_outputs
    assert str(encoder[2].size()) == 'torch.Size([3, 1, 10])'
    # encoder_hidden
    assert str(encoder[3].size()) == 'torch.Size([1, 1, 10])'
    

@pytest.fixture
def attn():
    attn = Attn(10, 7)
    encoder_outputs = Variable(torch.rand(3, 1, 10))
    hidden = Variable(torch.rand(1, 1, 10))
    energy = attn(hidden, encoder_outputs)
    return energy

def testAttn(attn):
    assert str(attn.size()) == 'torch.Size([1, 1, 3])'
    

@pytest.fixture
def decoder():
    # encoder 部分
    encoder = EncoderRNN(10, 10, 2)
    encoder_hidden = encoder.init_hidden()
    word_input = Variable(torch.LongTensor([[1, 2, 3]]))
    encoder_outputs, encoder_hidden = encoder(word_input, encoder_hidden)
    
    
    decoder = DecoderRNN(10, 10)
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(1, 1, decoder.hidden_size))
    word_inputs = Variable(torch.LongTensor([[5, 6, 9]]))
    
    for i in range(3):
        decoder_output, decoder_context, decoder_hidden, decoder_attn  = decoder(word_inputs[0][i].view(1, -1), decoder_context, decoder_hidden, encoder_outputs)
    
    return decoder_output, decoder_context, decoder_hidden, decoder_attn
    

def testDecoderRNN(decoder):
    # decoder_output
    assert str(decoder[0].size()) == 'torch.Size([1, 1, 10])'
    # decoder_context
    assert str(decoder[1].size()) == 'torch.Size([1, 1, 10])'
    # decoder_hidden
    assert str(decoder[2].size()) == 'torch.Size([1, 1, 10])'
    # decoder_attn
    assert str(decoder[3].size()) == 'torch.Size([1, 1, 3])'
    
    
    