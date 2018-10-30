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
import time
from PIL import Image, ImageDraw

def make_feed_dict(network, batch_size, img_arr, rect_labels, rects, pos_th, neg_th):
    feed_dict = {}
    
    label_dict = {}
    for i in range(2, 5 + 1):
        anchor_ph, anchor = network.get_anchor("reg_label{}".format(i))
        feed_dict[anchor_ph] = anchor.reshape(batch_size,
                                              anchor.shape[0],
                                              anchor.shape[1],
                                              anchor.shape[2],
                                              anchor.shape[3])

        cls, reg = encode_anchor_label(rect_labels, rects, anchor.reshape(-1, 4), pos_th, neg_th)
        tgt_layer_shape = (network.get_layer("cls{}".format(i)).get_shape().as_list())
        label_dict["cls_label{}".format(i)] = cls.reshape(batch_size, tgt_layer_shape[1], tgt_layer_shape[2], -1)
        label_dict["reg_label{}".format(i)] = reg.reshape(batch_size, tgt_layer_shape[1], tgt_layer_shape[2], -1, 4)
    feed_dict.update(network.create_feed_dict(input_image = img_arr.reshape(-1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]),
                                              label_dict = label_dict,
                                              is_training = True))
    
    return feed_dict


def focal_trial():
    
    anchor_size = 2.0 ** np.arange(0.0, 1.0, 0.25)
    anchor_asp  = np.linspace(0.5, 2.0, 3)
    img_h  = 256
    img_w  = img_h * 2
    img_ch = 3
    
    tgt_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"]]
    
    #from PIL import Image;Image.fromarray(rgb_arr_).show();exit()
    network = ImageNetwork(image_h  = img_h,
                           image_w  = img_w,
                           image_ch = img_ch)
    
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(7, 7, 2, 2, True), 32, "relu")
    
    layer = network.get_input(name = None)
    while layer.get_shape().as_list()[1] > (2 ** 5):
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 64, "relu")
        layer = network.get_input(name = None)

    # conv1_x
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 2, 2, True), 32, "relu", name = "e1_1")
    
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
        network.add_reshape(shape = [-1,
                                     int(network.get_input(None).get_shape().as_list()[1]),
                                     int(network.get_input(None).get_shape().as_list()[2]),
                                     anchor_size.size * anchor_asp.size,
                                     1 + len(tgt_words_list)])
        network.add_softmax(name = "cls{}".format(i))
        # regression
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu", input_name = feature_layer_name)
        for l in range(4 - 1 - 1):
            network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 256, "relu")
        network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), anchor_size.size * anchor_asp.size * 4, "relu")
        network.add_reshape(shape = [-1,
                                     network.get_input(None).get_shape().as_list()[1],
                                     network.get_input(None).get_shape().as_list()[2],
                                     anchor_size.size * anchor_asp.size,
                                     4],
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
    epoch_num = 1000
    lr = 1e-5
    
    bdd = BDD100k()
    bdd = BDD100k(resized_h = img_h,
                  resized_w = img_w)
    total_loss = network.get_total_loss()
    optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss)
    
    train_type = "val"
    val_type = "val"
    if os.name == "nt":
        train_type = "debug"
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for epoch in range(epoch_num):
            batch_cnt = 0
            for b in tqdm(range(bdd.get_sample_num(train_type) // batch_size)):
                batch_cnt += batch_size
                
                # one image
                rect_labels = np.empty(0)
                img_arr, rect_labels, rects, _1, _2 = bdd.get_vertices_data(train_type, tgt_words_list)
                learn_feed_dict = make_feed_dict(network, batch_size, img_arr, rect_labels, rects, pos_th = 0.5, neg_th = 0.4)
                
                if 0:
                    # visualize annotation rect
                    pil = Image.fromarray(img_arr.astype(np.uint8))
                    drw = ImageDraw.Draw(pil)
                    for rect in rects:
                        drw.rectangle((rect[1] * img_w,
                                       rect[0] * img_h,
                                       rect[3] * img_w,
                                       rect[2] * img_h),
                                      outline = (255,0,0))
                    pil.show()
                    #exit()
                
                if 0:
                    pal = []
                    pal.append((0,0,255))
                    pal.append((255,0,0))
                    pil_img = Image.fromarray(img_arr.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_img)
                    # visualize anchored label
                    for i in range(2, 5 + 1):
                        cls = sess.run(network._ImageNetwork__label_dict["cls_label{}".format(i)], feed_dict = learn_feed_dict)
                        reg = make_anchor(network.get_layer("reg{}".format(i)).get_shape().as_list()[1:1+2], size_list = anchor_size, asp_list = anchor_asp)
                        cls = cls.flatten()
                        reg = reg[cls > 0]
                        cls = cls[cls > 0]
                        for j in range(cls.size):
                            draw.rectangle((reg[j][1] * img_w,
                                            reg[j][0] * img_h,
                                            reg[j][3] * img_w,
                                            reg[j][2] * img_h),
                                            outline = pal[cls[j] - 1])
                    pil_img.show()
                    exit()

                if 0:
                    pal = []
                    pal.append((0,0,255))
                    pal.append((255,0,0))
                    pil_img = Image.fromarray(img_arr.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_img)
                    # visualize anchored label
                    for i in range(2, 5 + 1):
                        cls = sess.run(network._ImageNetwork__label_dict["cls_label{}".format(i)], feed_dict = learn_feed_dict)
                        reg = sess.run(network._ImageNetwork__label_dict["reg_label{}".format(i)], feed_dict = learn_feed_dict)
                        reg = reg[cls > 0]
                        cls = cls[cls > 0]
                        for j in range(cls.size):
                            draw.rectangle((reg[j][1] * img_w,
                                            reg[j][0] * img_h,
                                            reg[j][3] * img_w,
                                            reg[j][2] * img_h),
                                            outline = pal[cls[j] - 1])
                    pil_img.show()
                    exit()

                sess.run(optimizer, feed_dict = learn_feed_dict)
                #learn_loss = sess.run(total_loss, feed_dict = learn_feed_dict))
                
            if (epoch % 10 == 0) and (epoch > 0):
                starttime = time.time()
                val_loss = 0.0
                for val_idx in tqdm(range(bdd.get_sample_num(val_type))):
                    # one image
                    img_arr, rect_labels, rects, _1, _2 = bdd.get_vertices_data(train_type, tgt_words_list, index = val_idx)
                    eval_feed_dict = make_feed_dict(network, batch_size, img_arr, rect_labels, rects, pos_th = 0.5, neg_th = 0.4)
                    one_loss = sess.run(total_loss, feed_dict = eval_feed_dict)
                    val_loss = val_loss + one_loss
                    
                    # output prediction image
                    pal = []
                    pal.append((0,0,255))
                    pal.append((255,0,0))
                    pil_img = Image.fromarray(img_arr.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_img)
                    for i in range(2, 5 + 1):
                        cls, score, rect = decode_anchor_prediction(anchor_cls = sess.run(network.get_layer("cls{}".format(i)), feed_dict = eval_feed_dict),
                                                                    anchor_reg_t = sess.run(network.get_layer("reg{}".format(i)), feed_dict = eval_feed_dict),
                                                                    size_list = anchor_size,
                                                                    asp_list = anchor_asp,
                                                                    thresh = 0.5)
                        for j in range(cls.size):
                            if cls[j] != 0:
                                draw.rectangle((rect[j][1] * img_w,
                                                rect[j][0] * img_h,
                                                rect[j][3] * img_w,
                                                rect[j][2] * img_h),
                                                outline = pal[cls[j] - 1])
                                draw.text((rect[j][1] * img_w, rect[j][0] * img_h),
                                          text = "{:.2f}".format(score[j]),
                                          fill = pal[cls[j] - 1])
                    dst_dir = os.path.join("result", "epoch{0:04d}".format(epoch))
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    dst_name = "img{0:05d}.png".format(val_idx)
                    dst_path = os.path.join(dst_dir, dst_name)
                    pil_img.save(dst_path)
                    
                endtime = time.time()
                val_log = "loss={}".format(val_loss) + "," + "{:.2f}sec".format(endtime-starttime)
                print(val_log)
                open("eval_log.txt", "a").write(val_log + "\n")
            
if "__main__" == __name__:
    focal_trial()