import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

k = 5
d = 5
matrix = torch.FloatTensor(k, d).random_(from=0, to=10000)
print (matrix)
