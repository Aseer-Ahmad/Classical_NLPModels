#!/usr/bin/env python3
"""
INSTRUCTIONS:
0. SSH into the GPU cluster:
ssh username@conduit.cs.uni-saarland.de
OR
ssh username@conduit2.cs.uni-saarland.de
1. Create the script file in your user directory i.e., `touch mnist.py`
2. Paste content of `mnist.py` i.e., `vim mnist.py`
3. Set full permissions on your script i.e., `chmod 777 mnist.py`
4. Alternatively, for steps 1-2, you can use a FTP client on port `22`, this will be useful when transferring
datasets to cluster.
5. Repeat steps 1-2 for creating job config `pt_mnist_docker.sub`
6. Submit your job, i.e., `condor_submit pt_mnist_docker.sub`
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import logging

#now we will create and configure logger
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='out.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

## load mnist dataset
use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

logger.info ('==>>> total trainning batch number: {}'.format(len(train_loader)))
logger.info ('==>>> total testing batch number: {}'.format(len(test_loader)))

## network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "mnist_lenet.ckpt"

## training
model = LeNet()

if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            logger.info (f'==>>> epoch: {epoch}, batch index: {batch_idx+1}, train loss: {ave_loss}')
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        if use_cuda:
            x, target = x.cuda(), target.cuda()
        with torch.no_grad():
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            # smooth average
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1

            if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
                logger.info (f'==>>> epoch: {epoch}, batch index: {batch_idx+1}, test loss: {ave_loss}, acc: {correct_cnt * 1.0 / total_cnt}')
torch.save(model.state_dict(), model.name())
