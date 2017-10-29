import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

#k = 5
#d = 5   
#matrix = Variable(torch.DoubleTensor(k, d).random_(1000))
#print (matrix)

def Determinant(matrix):
    m, _ = matrix.size()
    if m==2:
#        print (matrix)
#        print (matrix[0, 0])
        result = matrix[0, 0]*(matrix[1, 1])-matrix[0, 1]*(matrix[1, 0])
#        print (result)
        return result
    else:
        summary = []
        for i in range(m):
            # pow((-1), i)
#            print (m)
#            print (matrix)
            if i==0:
                matrix_sub = matrix[1:, 1:]
            elif i==m-1:
                matrix_sub = matrix[1:, :-1]
            else:
                matrix_sub = torch.cat([matrix[1:, :i], matrix[1:, i+1:]], -1)
#            print (m)
#            print (matrix_sub, 'matrix_sub')
#            print ((pow((-1), i)*Determinant(matrix_sub)))
            summary.append(pow((-1), i)*Determinant(matrix_sub)*matrix[0, i])
        summary = torch.stack(summary)
#        print (summary)
        summary = torch.sum(summary)
        return summary

#print (Determinant(matrix))
