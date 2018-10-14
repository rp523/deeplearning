#coding: utf-8
import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../")
from dataset.mnist import MNIST
from model.network import ImageNetwork
from transformer import *
from trainer import Trainer

def mnist_trial():
    
    mnist = MNIST()

    network = ImageNetwork(image_h = 28,
                           image_w = 28,
                           image_ch = 1)
    network.add_conv(ImageNetwork.FilterParam(3, 3, 1, 1, True), 32)
    network.add_batchnorm()
    network.add_activation("relu")
    network.add_pool("MAX", ImageNetwork.FilterParam(2, 2, 2, 2, True))
    network.add_conv(ImageNetwork.FilterParam(3, 3, 1, 1, True), 64)
    network.add_batchnorm()
    network.add_activation("relu")
    network.add_pool("MAX", ImageNetwork.FilterParam(2, 2, 2, 2, True))
    network.add_full_connect(1024)
    network.add_activation("relu")
    #network.add_dropout(0.5)
    network.add_full_connect(10)
    network.add_softmax("answer")
    network.add_loss("cross_entropy", name = "ce_loss")
    #network.show()
    
    batch_size = 64
    epoch_num = 100
    lr = 1e-2
    x, y = mnist.get_data("train")
    y = transform_one_hot(y, 10)
    xv, yv = mnist.get_data("val")
    trainer = Trainer(model = network,
                      x = x,
                      y_dict = {"ce_loss":y})
    
    for epoch in range(epoch_num):
        batch_cnt = 0
        for b in (range(x.shape[0] // batch_size)):
            batch_cnt += batch_size
            trainer.training(lr = lr,
                             batch_size = batch_size,
                             valid_loss_dict = None)
            if b % 100 == 0:
                ans_mat = trainer.predict(xv, ["answer"])[0]
                ans = np.argmax(ans_mat, axis = 1)
                acc = np.average(ans == yv)
                print("[epoch={e}/{et}][batch={b}/{bt}] acc={acc}".format(e = epoch, et = epoch_num, b = batch_cnt, bt = x.shape[0], acc = acc))
