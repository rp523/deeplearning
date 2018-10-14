#coding: utf-8
import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../")
from dataset.bdd100k import BDD100k
from model.network import ImageNetwork
from transformer import *
from trainer import Trainer

def focal_trial():
    
    bdd = BDD100k(4, 4)
    tgt_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"]]
    
    rgb_arr_, _1, _2, _3, _4 = bdd.get_vertices_data("val", tgt_words_list, 0)
    network = ImageNetwork(image_h  = rgb_arr_.shape[0],
                           image_w  = rgb_arr_.shape[1],
                           image_ch = rgb_arr_.shape[2])
    
    # conv1_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(7, 7, 2, 2, True), 32, "relu")

    # conv2_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 64, "relu")
    for i in range(3):
        network.add_branching("conv2_{x}".format(x = i + 1))
        for j in range(2):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 64, "relu")
        network.add_injection(injection_name = "conv2_{x}".format(x = i + 1))
    
    # conv3_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 128, "relu")
    for i in range(4):
        network.add_branching("conv3_{x}".format(x = i + 1))
        for j in range(2):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 128, "relu")
        network.add_injection(injection_name = "conv3_{x}".format(x = i + 1))

    # conv4_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 256, "relu")
    for i in range(6):
        network.add_branching("conv4_{x}".format(x = i + 1))
        for j in range(2):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu")
        network.add_injection(injection_name = "conv4_{x}".format(x = i + 1))
    
    # conv5_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 512, "relu")
    for i in range(4):
        network.add_branching("conv5_{x}".format(x = i + 1))
        for j in range(2):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 512, "relu")
        network.add_injection(injection_name = "conv5_{x}".format(x = i + 1))

    network.add_softmax("answer")
    network.add_loss("cross_entropy", name = "ce_loss")
    network.show();exit()
    
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
