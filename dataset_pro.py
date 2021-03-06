import os
import pickle

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
# from transforms import Scale
import os
from scipy.misc import imread, imsave, imresize
import scipy.io as sio

class miniImagenet(Dataset):
    def __init__(self, split='train', way=5, shot=5, quiry=15):
        # with open('label2dir_train.pkl', 'rb') as f:
        #     # split: 'train', 'val', 'test'
        #     self.label2dir_train = pickle.load(f)
        # with open('label2dir_val.pkl', 'rb') as f:
        #     # split: 'train', 'val', 'test'
        #     self.label2dir_val = pickle.load(f)
        # with open('label2dir_test.pkl', 'rb') as f:
        #     # split: 'train', 'val', 'test'
        #     self.label2dir_test = pickle.load(f)
        with open('label2dir_' + split + '.pkl', 'rb') as f:
            # split: 'train', 'val', 'test'
            self.label2dir = pickle.load(f)
        self.way = way
        self.shot = shot
        self.quiry = quiry
        self.transform_aug_1 = transforms.Compose([
                                # Scale([84, 84]),
                                transforms.Pad(4),
                                transforms.RandomCrop([84, 84]),
                                # transforms.RandomHorizontalFlip(),
                                ])
        self.transform_aug_2 = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                #                     std=[0.5, 0.5, 0.5]),
                                ])

        self.if_aug = (split=='train')
        if split=='train':
            self.setsize = 64
        if split=='val':
            self.setsize = 16
        if split=='test':
            self.setsize = 20

        self.split = split

    def __getitem__(self, index):

        # dir_name = '../dataset/mini'
        support_list = [] # way*shot
        sample_list = [] # quiry
        index_c = random.sample(range(self.setsize), self.way)
        support_label = []
        sample_label = []
        # print (index_c)
        # print (label)
        for c in range(self.way):
            index_i = random.sample(range(600), self.shot+self.quiry) # shuffled
            # print (index_i)
            for i in range(self.shot+self.quiry):
                string = "%02d" % index_c[c] + "%03d" % index_i[i]
                if i<self.shot:
                    support_list.append(self.label2dir[string])
                    support_label.append(c)
                else:
                    sample_list.append(self.label2dir[string])
                    sample_label.append(c)

        # print (support_list.__len__()) # 25
        # print (sample_list.__len__()) # 5
        support_img = [] # [way, shot, 84, 84, 3]
        sample_img = [] # [quiry, 84, 84, 3]
        for imgfile in support_list:
            img = Image.open(imgfile).convert('RGB')
            if self.if_aug:
                img = self.augmentation(img)
                img = self.transform_aug_1(img)
            img = self.transform_aug_2(img)
            support_img.append(img)
        for imgfile in sample_list:
            img = Image.open(imgfile).convert('RGB')
            if self.if_aug:
                img = self.transform_aug_1(img)
            img = self.transform_aug_2(img)
            sample_img.append(img)
        support_img = torch.stack(support_img)
        sample_img = torch.stack(sample_img)
        support_img = support_img.view(self.way, self.shot, 3, 84, 84)
        sample_img = sample_img.view(self.way, self.quiry, 3, 84, 84)

        sample_label = torch.LongTensor(sample_label)
        support_label = torch.LongTensor(support_label)
        # print (support_label)
        # print (sample_label)
        '''one hot'''
        # support_label = support_label.unsqueeze(-1)
        # sample_label = sample_label.unsqueeze(-1)
        # support_label = torch.zeros(self.way, self.way).scatter_(1, support_label, 1)
        # sample_label = torch.zeros(self.way, self.way).scatter_(1, sample_label, 1)
        # print (support_label.size())
        # support_label = support_label.unsqueeze(1).repeat(1, self.shot, 1)# [self.way*self.shot, self.way]
        # sample_label = sample_label.unsqueeze(1).repeat(1, self.quiry, 1)
        # print (support_label) # way, shot, way
        # print (sample_label) # way, quiry, way

        return support_img, sample_img, sample_label

    def augmentation(self, img):
        ''' ratotion n*90'''
        angle = random.randint(0, 3)*90
        img = img.rotate(angle)
        ''' ratotion -5~5 '''
        angle = random.random()*10.0 - 5.0
        img = img.rotate(angle)
        ''' left right '''
        flag = random.randint(0, 1)
        if flag==1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # # top buttom
        # flag = random.randint(0, 1)
        # if flag==1:
        #     img = img.transpose(Image.FLIP_UP_BOTTOM)
        ''' contrast '''
        contrast = random.random()*0.5 - 0.25 + 1.0
        img = ImageEnhance.Contrast(img).enhance(contrast)
        # brightness
        bright = random.random()*0.5 - 0.25 + 1.0
        img = ImageEnhance.Brightness(img).enhance(bright)
        # sharpness
        sharpness = random.random()*0.5 - 0.25 + 1.0
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        # color
        color = random.random()*0.5 - 0.25 + 1.0
        img = ImageEnhance.Color(img).enhance(color)

        return img

    def __len__(self):
        return int(self.setsize*600/self.quiry)
        # return len(self.label2dir)

#transform1 =
#transform2 = transforms.Compose([
#    transforms.Scale([128, 128]),
#    transforms.Pad(4),
#    transforms.RandomCrop([128, 128]),
##    transforms.ToTensor(),
##    transforms.Normalize(mean=[0.5, 0.5, 0.5],
##                        std=[0.5, 0.5, 0.5])
#])


def collate_data(batch):

    support_imgs, sample_imgs, labels = [], [], []
    batch_size = len(batch)

    for i, b in enumerate(batch):
        support_img, sample_img, label = b
        support_imgs.append(support_img)
        sample_imgs.append(sample_img)
        labels.append(label)

    return torch.stack(support_imgs), torch.stack(sample_imgs), torch.stack(labels)
