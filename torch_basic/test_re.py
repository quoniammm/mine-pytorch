import torch
from torch.autograd import Variable

import pytest
from NMT_re import EncoderRNN

@pytest.fixture
def encoder():
    encoder = EncoderRNN(10, 10, 3)
    # [1, 3]
    word_inputs = Variable(torch.LongTensor([[2, 5, 7]]))
    # [3, 1, 10]
    hidden = encoder.init_hidden()
    # embedded [1, 1, 10]
    # [3, 1, 10]
    encoder_output, encoder_hidden = encoder(word_inputs, hidden)
    return encoder_output, encoder_hidden

def test_encoder_rnn(encoder):
    assert str(encoder[0].size()) == 'torch.Size([3, 1, 10])'
    assert str(encoder[1].size()) == 'torch.Size([3, 1, 10])'