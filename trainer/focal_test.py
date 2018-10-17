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
    img_h  = 128
    img_w  = 256
    img_ch = 3
    
    tgt_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"]]
    
    #from PIL import Image;Image.fromarray(rgb_arr_).show();exit()
    network = ImageNetwork(image_h  = img_h,
                           image_w  = img_w,
                           image_ch = img_ch)
    
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

    for i in range(2, 5 + 1):
        # classificatin
        feature_layer_name = "c{}".format(i)
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu", input_name = feature_layer_name)
        for l in range(4 - 1 - 1):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu")
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), anchor_size.size * anchor_asp.size * (1 + len(tgt_words_list)), "relu")
        network.add_softmax(name = "cls{}".format(i))
        # regression
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu", input_name = feature_layer_name)
        for l in range(4 - 1 - 1):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu")
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), anchor_size.size * anchor_asp.size * 4, "relu",
                                       name = "reg{}".format(i))
        network.add_rect_loss(name = "loss{}".format(i),
                              gamma = 2.0,
                              size_list = anchor_size,
                              asp_list = anchor_asp,
                              cls_layer_name = "cls{}".format(i),
                              reg_layer_name = "reg{}".format(i),
                              cls_label_name = "cls_label{}".format(i),
                              reg_label_name = "reg_label{}".format(i))
    batch_size = 1
    epoch_num = 100
    
    bdd = BDD100k()
    bdd = BDD100k(resized_h = img_h,
                  resized_w = img_w)
    total_loss = network.get_total_loss()
    trainer = Trainer(model = network,
                      total_loss = total_loss,
                      lr = 1e-5)
    
    for epoch in range(epoch_num):
        batch_cnt = 0
        for b in (range(bdd.get_sample_num("val") // batch_size)):
            batch_cnt += batch_size
            img_arr, rect_labels, rects, _1, _2 = bdd.get_vertices_data("val", tgt_words_list)
            label_dict = {}
            for i in range(2, 5 + 1):
                anchor_ph, anchor = network.get_anchor("reg_label{}".format(i))
                label_dict["reg_label{}".format(i)] = #TODO
                label_dict["cls_label{}".format(i)] = encode_anchor_label(rect_labels, rects, anchor.reshape(-1, 4), 0.5, 0.4)
            feed_dict = network.create_feed_dict(input_image = img_arr.reshape(-1, img_h, img_w, img_ch),
                                                 label_dict = label_dict,
                                                 is_training = True)
            for i in range(2, 5 + 1):
                anchor_ph, anchor = network.get_anchor("reg_label{}".format(i))
                feed_dict[anchor_ph] = anchor.reshape(batch_size,
                                                      anchor.shape[0],
                                                      anchor.shape[1],
                                                      anchor.shape[2],
                                                      anchor.shape[3])
            for k, v in feed_dict.items():
                print(k)
            trainer.training(feed_dict = feed_dict)
            print("[epoch={e}/{et}][batch={b}/{bt}]".format(e = epoch,
                                                            et = epoch_num,
                                                            b = batch_cnt,
                                                            bt = network.get_sample_num("train")))

if "__main__" == __name__:
    focal_trial()