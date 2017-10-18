import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import jieba
import math
import time
from collections import Counter

path = '../data/cmn-eng'

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

# 

# 