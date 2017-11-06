import pickle
import os
from PIL import Image
from collections import Counter
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
#from tqdm import tqdm
import time
import random
import csv
import datetime
from dataset_pro import miniImagenet, collate_data
from model_pro import ProtoNetwork

# Training settings
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--n-epoch', type=int, default=10000, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--is-multi-gpu', default=False, help='if use multiple gpu(default: False)')
parser.add_argument('--if-train', default=True)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--test-interval', type=int, default=1)
parser.add_argument('--from-epoch', type=int, default=35)
parser.add_argument('--nchannel', type=int, default=128)
parser.add_argument('--way', type=int, default=5)
parser.add_argument('--shot', type=int, default=5)
parser.add_argument('--quiry', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def train(epoch):
    train_set = DataLoader(miniImagenet('train', way=args.way, shot=args.shot, quiry=args.quiry),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=16,
                    collate_fn=collate_data,
                    pin_memory=args.cuda)

    protonet.train(True)
    avg_loss = 0
    avg_acc = 0

    for step, (support, sample, label) in enumerate(train_set):

        # start_time = time.time()
        if args.cuda:
            support, sample, label = \
            Variable(support).cuda(), Variable(sample).cuda(), Variable(label).cuda()

        else:
            support, sample, label = \
            Variable(support), Variable(sample), Variable(label)

        protonet.zero_grad()
        output, softmax = protonet(support, sample)
        # print (output)
        # print (label)
        output = output.view(-1, args.way)
        # print (output)
        # print (softmax)
        # print (label)
        label = label.view(-1)
        pred_label = softmax.data.cpu().numpy().argmax(1)
        # print (pred_label)
        accuracy = np.mean(label.data.cpu().numpy() == pred_label)
        avg_acc += accuracy
        # print (output.size())
        # print (label.size())
        loss = torch.sum(criterion(output, label))
        loss.backward()
        optimizer.step()

        avg_loss += loss.data[0]

        # exm_per_sec = args.batch_size / (time.time() - start_time)
        if step % args.log_interval == 0:
            print ('{}; <Train> Epoch: {}; Step: {:d}; Loss: {:.10f}; Avg_loss: {:.10f}; Avg_accuracy: {:.10f}' \
                   .format(datetime.datetime.now(), epoch, step, loss.data[0], avg_loss/(step+1), avg_acc/(step+1)))

    with open('logs_train_pro.csv', 'a') as csvfile_train:
        fieldnames_train = ['epoch', 'train_loss', 'train_acc']
        writer_train = csv.DictWriter(csvfile_train, fieldnames=fieldnames_train)
        writer_train.writerow({'epoch':epoch, 'train_loss':avg_loss/(step+1), 'train_acc':avg_acc/(step+1)})
    print (' ')


def val(epoch):
    train_set = DataLoader(miniImagenet('val', way=args.way, shot=args.shot, quiry=args.quiry),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=16,
                    collate_fn=collate_data,
                    pin_memory=args.cuda)

    protonet.train(False)
    avg_loss = 0
    avg_acc = 0

    for step, (support, sample, label) in enumerate(train_set):

        # start_time = time.time()
        if args.cuda:
            support, sample, label = \
            Variable(support).cuda(), Variable(sample).cuda(), Variable(label).cuda()

        else:
            support, sample, label = \
            Variable(support), Variable(sample), Variable(label)

        protonet.zero_grad()
        output, softmax = protonet(support, sample)

        output = output.view(-1, args.way)
        label = label.view(-1)
        pred_label = softmax.data.cpu().numpy().argmax(1)
        accuracy = np.mean(label.data.cpu().numpy() == pred_label)
        avg_acc += accuracy
        # print (output.size())
        # print (label.size())
        loss = torch.sum(criterion(output, label))
        # loss.backward()
        # optimizer.step()

        avg_loss += loss.data[0]

        # exm_per_sec = args.batch_size / (time.time() - start_time)
        if step % args.log_interval == 0:
            print ('{}; <Val> Epoch: {}; Step: {:d}; Loss: {:.10f}; Avg_loss: {:.10f}; Avg_accuracy: {:.10f}' \
                   .format(datetime.datetime.now(), epoch, step, loss.data[0], avg_loss/(step+1), avg_acc/(step+1)))

    with open('logs_val_pro.csv', 'a') as csvfile_train:
        fieldnames_train = ['epoch', 'val_loss', 'val_acc']
        writer_train = csv.DictWriter(csvfile_train, fieldnames=fieldnames_train)
        writer_train.writerow({'epoch':epoch, 'val_loss':avg_loss/(step+1), 'val_acc':avg_acc/(step+1)})
    print (' ')

def test(epoch):
    train_set = DataLoader(miniImagenet('test', way=args.way, shot=args.shot, quiry=args.quiry),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=16,
                    collate_fn=collate_data,
                    pin_memory=args.cuda)

    protonet.train(False)
    avg_loss = 0
    avg_acc = 0

    for step, (support, sample, label) in enumerate(train_set):

        # start_time = time.time()
        if args.cuda:
            support, sample, label = \
            Variable(support).cuda(), Variable(sample).cuda(), Variable(label).cuda()

        else:
            support, sample, label = \
            Variable(support), Variable(sample), Variable(label)

        protonet.zero_grad()
        output, softmax = protonet(support, sample) # batchsize, quiry, way

        output = output.view(-1, args.way)
        label = label.view(-1)
        pred_label = softmax.data.cpu().numpy().argmax(1)
        accuracy = np.mean(label.data.cpu().numpy() == pred_label)
        avg_acc += accuracy
        # print (output.size())
        # print (label.size())
        loss = torch.sum(criterion(output, label))# output: -dists=similarities, before softmax
        # loss.backward()
        # optimizer.step()
        avg_loss += loss.data[0]

        # exm_per_sec = args.batch_size / (time.time() - start_time)
        if step % args.log_interval == 0:
            print ('{}; <Test> Epoch: {}; Step: {:d}; Loss: {:.10f}; Avg_loss: {:.10f}; Avg_accuracy: {:.10f}' \
                   .format(datetime.datetime.now(), epoch, step, loss.data[0], avg_loss/(step+1), avg_acc/(step+1)))

    with open('logs_test_pro.csv', 'a') as csvfile_train:
        fieldnames_train = ['epoch', 'test_loss', 'test_acc']
        writer_train = csv.DictWriter(csvfile_train, fieldnames=fieldnames_train)
        writer_train.writerow({'epoch':epoch, 'test_loss':avg_loss/(step+1), 'test_acc':avg_acc/(step+1)})
    print (' ')


if args.is_multi_gpu:
    protonet = nn.parallel.DataParallel(ProtoNetwork(way=args.way, shot=args.shot, quiry=args.quiry, nchannel=args.nchannel))
else:
    protonet = ProtoNetwork(way=args.way, shot=args.shot, quiry=args.quiry, nchannel=args.nchannel)

if args.cuda:
    protonet = protonet.cuda()

if args.cuda:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(protonet.parameters(), lr=args.lr)

# model restore
try:
    checkpoint = torch.load('../ori/model/epoch_' + str(args.from_epoch) + '.pth')
    protonet.load_state_dict(checkpoint)
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")

# keep logs
csvfilename_train = 'logs_train_pro.csv'
csvfilename_test = 'logs_test_pro.csv'

with open('logs_train_pro.csv', 'a') as csvfile_train:
    fieldnames_train = ['epoch', 'train_loss', 'train_acc']
    writer_train = csv.DictWriter(csvfile_train, fieldnames=fieldnames_train)
    writer_train.writeheader()
with open('logs_val_pro.csv', 'a') as  csvfile_test:
    fieldnames_test = ['epoch', 'val_loss', 'val_acc']
    writer_test = csv.DictWriter(csvfile_test, fieldnames=fieldnames_test)
    writer_test.writeheader()
with open('logs_test_pro.csv', 'a') as  csvfile_test:
    fieldnames_test = ['epoch', 'test_loss', 'test_acc']
    writer_test = csv.DictWriter(csvfile_test, fieldnames=fieldnames_test)
    writer_test.writeheader()

for epoch in range(args.n_epoch):

    if args.if_train:
        train(epoch)
    # test(epoch)
    if epoch%args.test_interval==0:
        val(epoch)
        test(epoch)
        torch.save(protonet.state_dict(), 'model_pro/epoch_{}.pth'.format(epoch))
