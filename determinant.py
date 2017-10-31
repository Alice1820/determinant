import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

# k = 5
# d = 5
# matrix = Variable(torch.DoubleTensor(2, k, d).random_(1000))
# print (matrix)

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

def Determinant_byBatch(matrix):
    batchsize, m, _ = matrix.size()
    if m==2:
        # print (matrix)
#        print (matrix[0, 0])
        result = matrix[:, 0, 0]*matrix[:, 1, 1]-matrix[:, 0, 1]*matrix[:, 1, 0]
        # print (result)
        return result
    else:
        summary = []
        for i in range(m):
            # pow((-1), i)
#            print (m)
            # print (matrix)
            if i==0:
                matrix_sub = matrix[:, 1:, 1:]
            elif i==m-1:
                matrix_sub = matrix[:, 1:, :-1]
            else:
                matrix_sub = torch.cat([matrix[:, 1:, :i], matrix[:, 1:, i+1:]], -1)
#            print (m)
#            print (matrix_sub, 'matrix_sub')
            coef = pow((-1), i)
            # print (coef*(Determinant_byBatch(matrix_sub)*matrix[:, 0, i]).unsqueeze(-1))
            summary.append(coef*(Determinant_byBatch(matrix_sub)*matrix[:, 0, i]).unsqueeze(-1))
        # print (summary)
        summary = torch.stack(summary, -1).view(batchsize, -1)
        # print (summary)
        summary = torch.sum(summary, -1)
        # print (summary)
        return summary

#print (Determinant(matrix))
# print (Determinant_byBatch(matrix))
