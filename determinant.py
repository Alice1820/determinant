import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

k = 5
d = 5
matrix = torch.LongTensor(k, d).random_()
print (matrix)
