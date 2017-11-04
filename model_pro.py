import torch
from torch import nn
from torch.nn.init import kaiming_uniform
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F
from determinant import Determinant, Determinant_byBatch

class MatchingNetwork(nn.Module):
    def __init__(self, way=5, shot=5, quiry=15):
        super(MatchingNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(64)

        self.way = way
        self.shot = shot
        self.quiry = quiry

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

    def simplex_distance_byBatch(self, support, sample):
        '''Calculate cosine distance among all k examples with respect to m sample
        Args:
        support: [batchsize, way*shot, D]
        sample: [batchsize, quiry, D]
        Returns:
        dists: [batchsize*way*shot, quiry] same for examples in a same category
        '''
        # print (support)
        # print (sample)
        '''Volume_A'''
        support = support.view(-1, self.shot, 1600) # batchsize*way, shot, D
        matrix_A = support[:, 1:].sub(support[:, 0].unsqueeze(1).repeat(1, self.shot-1, 1)) # batchsize*way, shot-1, D
        matrix_A_trans = matrix_A.permute(0, 2, 1) # batchsize*way, D, shot-1
        matrix_A = matrix_A.bmm(matrix_A_trans) # batchsize*way, shot-1, shot-1
        # print (matrix_A)
        Volume_A = Determinant_byBatch(matrix_A) # batchsize*way
        # print (Volume_A)
        Volume_A = Volume_A.view(-1, self.way) # batchsize, way
        # print (matrix_A.size(), 'matrix_A.size()')
        '''Volume_B'''
        support = support.view(-1, self.way*self.shot, 1600)
        matrix_B = support.unsqueeze(1).repeat(1, self.quiry, 1, 1).sub(sample.unsqueeze(2).repeat(1, 1, self.way*self.shot, 1))
        # batchsize, quiry, way*shot, D
        # print (matrix_B.size(), 'matrix_B.size()')
        matrix_B = matrix_B.view(-1, self.shot, 1600) # batchsiz*quiry*way, shot, D
        matrix_B_trans = matrix_B.permute(0, 2, 1) # batchsiz*quiry*way, D, shot
        matrix_B = matrix_B.bmm(matrix_B_trans) # batchsiz*quiry*way, shot, shot
        # print (matrix_B)
        Volume_B = Determinant_byBatch(matrix_B) # batchsize*quiry*way
        # print (Volume_B)
        Volume_B = Volume_B.view(-1, self.quiry, self.way) # batchsize, quiry, way

        dists = Volume_B/(Volume_A.unsqueeze(1).repeat(1, self.quiry, 1))  # batchsize, quiry, way
        similarities = dists * (-1)
        # # dists = dists.unsqueeze(2).repeat(1, 1, self.shot, 1).view(-1, self.way*self.shot, self.quiry) # batchsize, way*shot, quiry
        # min_values, _ = torch.min(dists, -1) # batchsize, self.quiry, 1
        # # print (min_values.size())
        # dists = min_values.repeat(1, 1, self.way)/dists # batchsize, quiry, way
        # # print (dists)
        # similarities = dists.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, self.shot, 1).view(-1, self.way*self.shot, self.quiry)
        # # batchsize, way*shot, quiry

        return similarities

    def simplex_distance(self, support, sample):
        '''Calculate cosine distance among all k examples with respect to m sample
        Args:
        support: [way*shot, D]
        sample: [quiry, D]
        Returns:
        dists: [way, shot, quiry] same for examples in a same category
        '''
        support = support.view(self.way, self.shot, 1600)
        dists = []
        for s in range(self.way):
            matrix_A = support[s, 1:].sub(support[s, 0].repeat(self.shot-1, 1)) # shot-1, D
            matrix_A_trans = matrix_A.permute(1, 0) # D, shot-1
            matrix_A = matrix_A.unsqueeze(0).bmm(matrix_A_trans.unsqueeze(0)).squeeze() # self.shot-1, self.shot-1
            Volume_A = Determinant(matrix_A) # 0
        #    print (Volume_A, 'Volume_A')
            for q in range(self.quiry):
                matrix_B = support[s].sub(sample[q].repeat(self.shot, 1))
            #    print (matrix)
                matrix_B_trans = matrix_B.permute(1, 0) # D, way*shot
                matrix_B = matrix_B.unsqueeze(0).bmm(matrix_B_trans.unsqueeze(0)).squeeze()
            #    print (matrix)
            #    print (Determinant(matrix))
                Volume_B = Determinant(matrix_B)
                dists.append(Volume_B / Volume_A)
        # dists = torch.stack(dists).squeeze().view(self.way, self.quiry).unsqueeze(1).repeat(1, self.shot, 1)
        # way, shot, quiry
        # print (dists)
        # print (torch.max(dists))
        '''numpy'''
        # max_values = torch.max(dists).data.cpu().numpy()[0]
        # print (max_values)
        '''repeat tensor'''
        # max_values = torch.max(dists).unsqueeze(0).unsqueeze(0).repeat(self.way, self.shot, self.quiry)

        # dists = dists / (max_values.repeat(1, self.way*self.shot))
        # dists = max_values / dists
        dists = dists.view(-1, self.quiry)

        return dists

    def AttentionalClassify_byBatch(self, similarities, support_set_y):
        """
        similarities: [batchsize, way*shot, quiry]
        support_set_y: [batchsize, way, shot, way] one hot
        """
        '''one_hot'''
        similarities = similarities.permute(0, 2, 1).contiguous().view(-1, self.way*self.shot) # [batchsize*quiry, way*shot]
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities) # batchsize*quiry, way*shot
        # print (softmax_similarities)
        # print (support_set_y)
        # softmax_similarities = 1 - softmax(similarities)
        softmax_similarities = softmax_similarities.view(-1, self.quiry, self.way*self.shot) # batchsize, self.quiry, self.way*self.shot
        preds = softmax_similarities.bmm(support_set_y.view(-1, self.way*self.shot, self.way)) # batchsize, quiry, way
        # print (preds)
        return preds

    def AttentionalClassify(self, similarities, support_set_y):
        """
        similarities: [batchsize, way*shot, quiry]
        support_set_y: [batchsize, way, shot, way] one hot
        """
        '''one_hot'''
        similarities = similarities.permute(0, 2, 1).contiguous().view(-1, self.way*self.shot) # [batchsize*quiry, way*shot]
        softmax = nn.Softmax()
        # print (similarities)
        max_values, _ = torch.max(similarities, dim=-1)
        # print (max_values)
        similarities = similarities/(max_values.repeat(1, self.way*self.shot))
        # print (similarities, 'similarities, normed')
        similarities = similarities.view(-1, self.way, self.shot)
        similarities = torch.sum(similarities, dim=2).view(-1, self.way) # batchsize, self.quiry, self.way
        # print (similarities)
        # print (softmax(similarities))
        softmax_similarities = softmax(similarities).view(-1, self.quiry, self.way) # batchsize, self.quiry, self.way
        preds = softmax_similarities.bmm(support_set_y.view(-1, self.way*self.shot, self.way)) # batchsize, quiry, way
        # print (preds)
        return preds

    def DistanceNetwork_byBatch(self, support, support_label, sample):
        '''make prediction with various distance
        Args:
        support: [batchsize, way*shot, D]
        support_label: [batchsize, way, shot, way]
        sample: [batchsize, quiry, D]
        Returns:
        prediction: [batchsize, quiry] 0~(way-1)
        '''
        batchsize, _, _ = sample.size()

        similarities = []
        # cosine_similarity = self.cosine_distance(support_set, sample_set) # [self.way*self.shot, self.quiry]
        simplex_similarities = self.simplex_distance_byBatch(support, sample) # batchsize, way*shot, quiry
        # similarities.append(cosine_similarity)
        # logits = self.AttentionalClassify_byBatch(simplex_similarities, support_label) # batchsize, quiry, way
        # print (logits)
        return simplex_similarities

    def DistanceNetwork(self, support, support_label, sample):
        '''make prediction with various distance
        Args:
        support: [batchsize, way*shot, D]
        support_label: [batchsize, way, shot, way]
        sample: [batchsize, quiry, D]
        Returns:
        prediction: [batchsize, quiry] 0~(way-1)
        '''
        # support = support.view(-1, self.way*self.shot, 1600) # batchsize*way*shot, 1600
        # sample = sample.view(-1, self.quiry, 1600) # batchsize*quiry, 1600
        # print (support.size())
        # print (sample.size())
        batchsize, _, _ = sample.size()

        similarities = []
        for b in range(batchsize):
            support_set = support[b] # 25, 1600
            sample_set = sample[b] # 15, 1600
            # cosine_similarity = self.cosine_distance(support_set, sample_set) # [self.way*self.shot, self.quiry]
            simplex_similarity = self.simplex_distance(support_set, sample_set) # way*shot, quiry
            # similarities.append(cosine_similarity)
            similarities.append(simplex_similarity)
        similarities = torch.stack(similarities) # batchsize, self.way*self.shot, self.quiry
        # similarities = similarities.view(self.way*self.shot, self.quiry)
        logits = self.AttentionalClassify(similarities, support_label) # batchsize, quiry, way
        # print (logits)
        return logits

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

    def forward(self, support, support_label, sample):
        # batchsize, way, shot, _, _, _ = support.size()
        # _, quiry, _, _, _ = sample.size()
        support = support.view(-1, 3, 84, 84)
        sample = sample.view(-1, 3, 84, 84)
        support = self.convnet(support) # [batchsize, way, shot, 1600]
        sample = self.convnet(sample) # [batchsize, quiry, 1600]
        # print (support.size())
        # print (sample.size())
        support = support.view(-1, self.way*self.shot, 1600)
        sample = sample.view(-1, self.quiry, 1600)
        logits = self.DistanceNetwork_byBatch(support, support_label, sample) # batchsize, wquiry, way
        # logits = self.DistanceNetwork(support, support_label, sample) # batchsize, quiry, way
        # print (logits.size())
        # print (sample_label.size())
        return logits
