import torch
from torch import nn
from torch.nn.init import kaiming_uniform
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F

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

    def simplex_distance(self, support, sample):
        '''Calculate cosine distance among all k examples with respect to m sample
        Args:
        support: [way*shot, D]
        sample: [quiry, D]
        Returns:
        dists: [way*shot, quiry] same for examples in a same category
        '''
        pass

    def AttentionalClassify(self, similarities, support_set_y):
        """
        similarities: [batchsize, way*shot, quiry]
        support_set_y: [batchsize, way*shot, way] one hot
        """
        '''one_hot'''
        similarities = similarities.permute(0, 2, 1).contiguous().view(-1, self.way*self.shot) # [batchsize, quiry, way*shot]
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities).view(-1, self.quiry, self.way*self.shot) # batchsize, self.quiry, self.way*self.shot
        preds = softmax_similarities.bmm(support_set_y) # batchsize, quiry, way
        return preds

    def DistanceNetwork(self, support, support_label, sample):
        '''make prediction with various distance
        Args:
        support: [batchsize, k, D]
        support_label: [batchsize, way*shot, way]
        sample: [batchsize, m, D]
        Returns:
        prediction: [batchsize] 0~(way-1)
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
            cosine_similarity = self.cosine_distance(support_set, sample_set) # [self.way*self.shot, self.quiry]
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities) # batchsize, self.way*self.shot, self.quiry
        # similarities = similarities.view(self.way*self.shot, self.quiry)
        logits = self.AttentionalClassify(similarities, support_label) # batchsize, quiry, way
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
        logits = self.DistanceNetwork(support, support_label, sample)
        # print (logits.size())
        # print (sample_label.size())
        return logits
