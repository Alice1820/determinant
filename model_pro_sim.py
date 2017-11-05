import torch
from torch import nn
from torch.nn.init import kaiming_uniform
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F
from determinant import Determinant, Determinant_byBatch

class ProtoNetwork(nn.Module):
    def __init__(self, way=5, shot=5, quiry=15, if_cuda=True):
        super(ProtoNetwork, self).__init__()

        self.if_cuda = if_cuda

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(64)

        self.softmax = nn.Softmax()
        self.way = way
        self.shot = shot
        self.quiry = quiry

        self.pdist = nn.PairwiseDistance(p=2)

    def cosine_distance(self, support, sample):
        '''Calculate cosine distance among all k examples with respect to m sample
        Args:
        support: [way*shot, D]
        sample: [quiry, D]
        Returns:
        dists: [way*shot, quiry]
        '''
        eps = 1e-10
        cosine_similarity = []
        # print (support.size(), 'support.size()')
        # print (sample.size(), 'sample.size()')
        for s in range(self.way*self.shot):
            for j in range(self.quiry):
                support_image = support[s]
                input_image = sample[j]
                # print (support_image.size())
                # print (input_image)
                sum_support = torch.sum(torch.pow(support_image, 2))
                support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
                '''bmm'''
                support_image = support_image.unsqueeze(0).unsqueeze(-1)
                input_image = input_image.unsqueeze(0).unsqueeze(0)
                # print (support_image.size())
                # print (input_image)
                dot_product = input_image.bmm(support_image).squeeze()
                cosine_similarity.append(dot_product * support_magnitude)
        cosine_similarity = torch.stack(cosine_similarity)
        cosine_similarity = cosine_similarity.view(self.way*self.shot, self.quiry)
        return cosine_similarity

    def euclidean_distance_byBatch(self, support, sample):
        '''Calculate cosine distance among all k examples with respect to m sample
        Args:
        support: [batchsize, way*shot, D]
        sample: [batchsize, way*quiry, D]
        Returns:
        dists: [batchsize, quiry, way] same for examples in a same category
        '''
        centers = torch.mean(support.view(-1, self.way, self.shot, 1600), 2) # batchsize, way, D
        # print (centers)
        centers = centers.squeeze().unsqueeze(1).repeat(1, self.way*self.quiry, 1, 1).view(-1, 1600) # batchsize, way*quiry, way, D
        sample = sample.unsqueeze(2).repeat(1, 1, self.way, 1).view(-1, 1600) # batchsize, way*quiry, way, D
        dists = self.pdist(centers, sample) # batchsize*quiry*way
        dists = dists.view(-1, self.way*self.quiry, self.way)
        dists = dists*(-1)

        return dists

    def simplex_distance_byBatch(self, support, sample):
        '''Calculate cosine distance among all k examples with respect to m sample
        Args:
        support: [batchsize, way*shot, D]
        sample: [batchsize, way*quiry, D]
        Returns:
        dists: [batchsize, quiry, way] same for examples in a same category
        '''
        batchsize, _, _ = support.size()
        I_B = Variable(torch.FloatTensor(np.eye(self.shot)))
        I_A = Variable(torch.FloatTensor(np.eye(self.shot-1)))
        if self.if_cuda:
            I_B = I_B.cuda()
            I_A = I_A.cuda()
        # print (support)
        # print (sample)
        '''Volume_A'''
        support = support.view(-1, self.shot, 1600) # batchsize*way, shot, D
        matrix_A = support[:, 1:].sub(support[:, 0].unsqueeze(1).repeat(1, self.shot-1, 1)) # batchsize*way, shot-1, D
        matrix_A_trans = matrix_A.permute(0, 2, 1) # batchsize*way, D, shot-1
        matrix_A = matrix_A.bmm(matrix_A_trans) + I_A.unsqueeze(0).repeat(batchsize*self.way, 1, 1)
        # matrix_A = matrix_A.bmm(matrix_A_trans)# batchsize*way, shot-1, shot-1
        # print (matrix_A)
        Volume_A = Determinant_byBatch(matrix_A) # batchsize*way
        # print (Volume_A)
        Volume_A = Volume_A.view(-1, self.way) # batchsize, way
        # print (matrix_A.size(), 'matrix_A.size()')
        '''Volume_B'''
        support = support.view(-1, self.way*self.shot, 1600)
        matrix_B = support.unsqueeze(1).repeat(1, self.way*self.quiry, 1, 1).sub(sample.unsqueeze(2).repeat(1, 1, self.way*self.shot, 1))
        # batchsize, quiry, way*shot, D
        # print (matrix_B.size(), 'matrix_B.size()')
        matrix_B = matrix_B.view(-1, self.shot, 1600) # batchsiz*quiry*way, shot, D
        matrix_B_trans = matrix_B.permute(0, 2, 1) # batchsiz*quiry*way, D, shot
        matrix_B = matrix_B.bmm(matrix_B_trans) + I_B.unsqueeze(0).repeat(batchsize*self.quiry*self.way, 1, 1) # batchsiz*quiry*way, shot, shot
        # print (matrix_B)
        # Volume_B = Determinant_byBatch(matrix_B) + I_B# batchsize*quiry*way
        Volume_B = Determinant_byBatch(matrix_B)
        # print (Volume_B)
        Volume_B = Volume_B.view(-1, self.way*self.quiry, self.way) # batchsize, quiry, way

        dists = Volume_B/(Volume_A.unsqueeze(1).repeat(1, self.way*self.quiry, 1))  # batchsize, quiry, way
        similarities = dists * (-1)

        return similarities

    def convnet(self, images):
        batch_size, img_size, _, _ = images.size()
        # print (images.size())
        """conv1"""
        x = self.conv1(images)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = F.max_pool2d(x, 2)
        """conv2"""
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = F.max_pool2d(x, 2)
        """conv3"""
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = F.max_pool2d(x, 2)
        """conv4"""
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        x_conv = F.max_pool2d(x, 2)
        """flatten"""
        x_flatten = x_conv.view(-1, 1600)
        return x_flatten

    def forward(self, support, sample):
        # batchsize, way, shot, _, _, _ = support.size()
        # _, quiry, _, _, _ = sample.size()
        support = support.view(-1, 3, 84, 84)
        sample = sample.view(-1, 3, 84, 84)
        support = self.convnet(support) # [batchsize, way, shot, 1600]
        sample = self.convnet(sample) # [batchsize, way, quiry, 1600]
        # print (support.size())
        # print (sample.size())
        support = support.view(-1, self.way*self.shot, 1600)
        sample = sample.view(-1, self.way*self.quiry, 1600)
        # logits = self.euclidean_distance_byBatch(support, sample) # [batchsize, quiry, way]
        logits = self.simplex_distance_byBatch(support, sample)
        softmax_logits = self.softmax(logits.view(-1, self.way)) # [batchsize, quiry, way]
        # logits = self.DistanceNetwork(support, support_label, sample) # batchsize, quiry, way
        # print (logits.size())
        # print (sample_label.size())
        return logits, softmax_logits
