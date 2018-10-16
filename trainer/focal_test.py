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
    tgt_words_list = [["car", "truck", "bus"], ["person", "rider"]]
    anchor_size = 2.0 ** np.arange(0.0, 1.0, 0.25)
    anchor_asp  = np.linspace(0.5, 2.0, 3)
    
    bdd = BDD100k(resized_h = 128,
                  resized_w = 256)
    tgt_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"]]
    
    rgb_arr_, _1, _2, _3, _4 = bdd.get_vertices_data("val", tgt_words_list, 1)
    #from PIL import Image;Image.fromarray(rgb_arr_).show();exit()
    network = ImageNetwork(image_h  = rgb_arr_.shape[0],
                           image_w  = rgb_arr_.shape[1],
                           image_ch = rgb_arr_.shape[2])
    
    # conv1_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(7, 7, 2, 2, True), 32, "relu", name = "e1_1")

    # conv2_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 64, "relu")
    for i in range(3):
        network.add_branching("e2_{x}".format(x = i + 1))
        for j in range(2):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 64, "relu")
        network.add_injection(injection_name = "e2_{x}".format(x = i + 1))
    
    # conv3_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 128, "relu")
    for i in range(4):
        network.add_branching("e3_{x}".format(x = i + 1))
        for j in range(2):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 128, "relu")
        network.add_injection(injection_name = "e3_{x}".format(x = i + 1))

    # conv4_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 256, "relu")
    for i in range(6):
        network.add_branching("e4_{x}".format(x = i + 1))
        for j in range(2):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu")
        network.add_injection(injection_name = "e4_{x}".format(x = i + 1))
    
    # conv5_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 256, "relu")
    for i in range(4):
        network.add_branching("e5_{x}".format(x = i + 1))
        for j in range(2):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu")
        network.add_injection(injection_name = "e5_{x}".format(x = i + 1))
    
    # top-bottom
    network.add_upsample(2, 2, name = "bottomup_5")
    network.add_conv(ImageNetwork.FilterParam(1, 1, 1, 1, True), 256, input_name = "e4_6")
    network.add_injection(injection_name = "bottomup_5", name = "c5")

    network.add_upsample(2, 2, name = "bottomup_4")
    network.add_conv(ImageNetwork.FilterParam(1, 1, 1, 1, True), 256, input_name = "e3_4")
    network.add_injection(injection_name = "bottomup_4", name = "c4")

    network.add_upsample(2, 2, name = "bottomup_3")
    network.add_conv(ImageNetwork.FilterParam(1, 1, 1, 1, True), 256, input_name = "e2_3")
    network.add_injection(injection_name = "bottomup_3", name = "c3")

    network.add_upsample(2, 2, name = "bottomup_2")
    network.add_conv(ImageNetwork.FilterParam(1, 1, 1, 1, True), 256, input_name = "e1_1")
    network.add_injection(injection_name = "bottomup_2", name = "c2")

    network.add_softmax("answer")
    network.add_loss("cross_entropy", name = "ce_loss")
    
    for i in range(2, 5 + 1):
        # classificatin
        feature_layer_name = "c{}".format(i)
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu", input_name = feature_layer_name)
        for l in range(4 - 1 - 1):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu")
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), anchor_size.size * anchor_asp.size * (1 + len(tgt_words_list)), "relu",
                                       name = "cls{}".format(i))
        # regression
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu", input_name = feature_layer_name)
        for l in range(4 - 1 - 1):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu")
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), anchor_size.size * anchor_asp.size * 4, "relu",
                                       name = "reg{}".format(i))
        terminal_shape = network.get_layer("reg{}".format(i)).get_shape().as_list()
        network.add_rect_loss(name = "loss{}".format(i),
                              gamma = 2.0,
                              cls_layer_name = "cls{}".format(i),
                              reg_layer_name = "reg{}".format(i),
                              cls_label_name = "cls_loss{}".format(i),
                              reg_label_name = "reg_loss{}".format(i))
    
    for i in range(2, 5 + 1):
        network.show_layer("cls{}".format(i))
        network.show_layer("reg{}".format(i))
    print("debug_done.");exit()
    anchor = make_anchor([terminal_shape[1], terminal_shape[2]], 
                                                   size_list = anchor_size,
                                                   asp_list = anchor_asp),
                              
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

if "__main__" == __name__:
    focal_trial()