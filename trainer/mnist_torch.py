#coding: utf-8
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.append("../")
from dataset.mnist import MNIST
import time 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sm = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sm(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def flatten(self, x):
        return x.view(-1, self.num_flat_features(x))
    
    
def train():
    device = torch.device("cuda:0")
    net = Net().to(device)
    mnist = MNIST()
    x, y = mnist.get_data("train")
    xv, yv = mnist.get_data("val")
    x  = np.transpose(x , (0, 3, 1, 2)) / 128 - 1
    xv = np.transpose(xv, (0, 3, 1, 2)) / 128 - 1
    x  = torch.from_numpy( x.astype(np.float32)).to(device)
    xv = torch.from_numpy(xv.astype(np.float32)).to(device)
    y  = torch.from_numpy( y.astype(np.int64)).to(device)
    yv = torch.from_numpy(yv.astype(np.int64)).to(device)
    
    EPOCH_NUM = 501
    BATCH_SIZE = 16
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr = 0.001,
                          momentum = 0.9,
                          weight_decay = 5e-5)
    running_loss = 0.0
    
    for epoch in range(EPOCH_NUM):
        t0 = time.time()
        running_loss = 0.
        for i, batch_try in enumerate(range(x.shape[0] // BATCH_SIZE)):
            batch_idx = np.random.choice(np.arange(x.shape[0]), BATCH_SIZE, replace = True)
            batch_x = x[batch_idx]
            batch_y = y[batch_idx]
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net.forward(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
        # print statistics
        t1 = time.time()
        
        with torch.no_grad():
            outputs = net.forward(xv)
            max_val, max_idx = outputs.max(1)
            correct = torch.eq(max_idx, yv).float().mean().cpu()
            correct = float(correct)
            print('[%.3f, %d, %5d] loss: %.3f acc: %.3f' %
                  ((t1 - t0), epoch + 1, i + 1, running_loss / 2000, correct * 100.0))
            
def main():
    train()

if "__main__" == __name__:
    main()
    print("Done.")