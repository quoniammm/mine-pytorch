import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

df_train_txt = pd.read_csv('training_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_train_var = pd.read_csv('training_variants')
df_test_txt = pd.read_csv('test_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_test_var = pd.read_csv('test_variants')
df_train = pd.merge(df_train_var, df_train_txt, how='left', on='ID')
df_test = pd.merge(df_test_var, df_test_txt, how='left', on='ID')
df_train.head()