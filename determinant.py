import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

k = 5
D = 5
matrix = torch.LongTensor(k, D).random_()
print (matrix)
