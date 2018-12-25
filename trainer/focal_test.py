#coding: utf-8
import os, sys, subprocess
import numpy as np
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../")

from dataset.bdd100k import BDD100k
from dataset.edgeai import EdgeAI

from model.network import ImageNetwork
from transformer import *
from trainer import Trainer
import time
from PIL import Image, ImageDraw
from datetime import datetime
import common.calc as calc

def make_dtc_feed_dict(network, batch_size,
                   img_arr,             # batch x h x w x ch
                   rect_labels_list,    # batch x label
                   rects_list,          # batch x label x 4
                   pos_th, neg_th):
    feed_dict = {}
    
    label_dict = {}
    for i in range(2, 5 + 1):
        anchor_ph, anchor = network.get_anchor("reg_label{}".format(i))
        anchor_ph_val = np.empty((batch_size,
                                  anchor.shape[0],
                                  anchor.shape[1],
                                  anchor.shape[2],
                                  anchor.shape[3])).astype(np.float)
        tgt_layer_shape = (network.get_layer("cls{}".format(i)).get_shape().as_list())
        cls_list = np.empty((batch_size, tgt_layer_shape[1] * tgt_layer_shape[2] * tgt_layer_shape[3])).astype(np.int)
        reg_list = np.empty((batch_size, tgt_layer_shape[1] * tgt_layer_shape[2] * tgt_layer_shape[3], 4)).astype(np.float)
        if (len(rect_labels_list) > 0) and (len(rects_list) > 0):
            for j, (rect_labels, rects) in enumerate(zip(rect_labels_list, rects_list)):
                anchor_ph_val[j] = anchor
                if (not rect_labels is None):
                    cls_list[j], reg_list[j] = encode_anchor_label(rect_labels, rects, anchor.reshape(-1, 4), pos_th, neg_th)
        label_dict["cls_label{}".format(i)] = cls_list.reshape(batch_size, tgt_layer_shape[1], tgt_layer_shape[2], -1)
        label_dict["reg_label{}".format(i)] = reg_list.reshape(batch_size, tgt_layer_shape[1], tgt_layer_shape[2], -1, 4)
        feed_dict[anchor_ph] = anchor_ph_val
    assert(img_arr.shape[0] == batch_size)
    feed_dict.update(network.create_feed_dict(input_image = img_arr,
                                              label_dict = label_dict,
                                              is_training = True))
    
    return feed_dict

def make_seg_feed_dict(network, batch_size,
                       img_arr,     # batch x img_h x img_w x ch
                       seg_arr):    # batch x h x w
    feed_dict = {}
    
    label_dict = {}
    seg_layer_shape = network.get_layer("seg").get_shape().as_list()
    seg_h = seg_layer_shape[1]
    seg_w = seg_layer_shape[2]
    cls_num = seg_layer_shape[3]
    label = np.zeros((batch_size, seg_h, seg_w, cls_num)).astype(np.float32)
    for b in range(batch_size):
        for cls in range(cls_num):
            shrink_seg = calc.thinout(seg_arr[b,:,:], seg_h, seg_w)
            label[b,:,:,cls] = 1 * (shrink_seg == cls)
    label_dict["seg_label"] = label
    feed_dict.update(network.create_feed_dict(input_image = img_arr,
                                              label_dict = label_dict,
                                              is_training = True))
    
    return feed_dict

def calc_class_freq(network, data, dat_type, tgt_words_list, reg_label_name_list, pos_th, neg_th):
    cls_num = len(tgt_words_list) + 1
    cnt = np.zeros(cls_num).astype(np.uint32)
    for b in tqdm(range(data.get_sample_num(dat_type))):
        _0, rect_labels, rects, _1, _2 = data.get_vertices_data(dat_type, tgt_words_list, index = b)
        for reg_label_name in reg_label_name_list:
            _, anchor = network.get_anchor(reg_label_name)
            cls, _ = encode_anchor_label(rect_labels, rects, anchor.reshape(-1, 4), pos_th, neg_th)
            for c in range(cnt.size):
                cnt[c] = cnt[c] + np.sum(cls == c)
    return cnt / np.sum(cnt)

def get_dtc_loss(network, weight_decay):
    loss = tf.constant(0.0, dtype = tf.float32)

    cls_loss = tf.constant(0.0, dtype = tf.float32)
    reg_loss = tf.constant(0.0, dtype = tf.float32)
    for k, v in network.get_loss_dict().items():
        # weighting rule below is cited from: https://arxiv.org/abs/1705.07115
        if k.find("cls") >= 0:
            cls_loss = cls_loss + v
        elif k.find("reg") >= 0:
            reg_loss = reg_loss + v
    cls_s = tf.Variable(0.0, dtype = tf.float32)
    reg_s = tf.Variable(0.0, dtype = tf.float32)
    loss = loss + (1.0 * tf.exp(- cls_s) * cls_loss + 0.5 * cls_s)
    loss = loss + (0.5 * tf.exp(- reg_s) * reg_loss + 0.5 * reg_s)
    
    for weight in network.get_weight_list():
        loss = loss + tf.cast(weight_decay, tf.float32) * 0.5 * tf.reduce_sum(tf.cast(weight, tf.float32) ** 2)
    return loss, [tf.exp(-cls_s), cls_loss, 0.5 * tf.exp(-reg_s), reg_loss]

def get_seg_loss(network, weight_decay):
    loss = tf.constant(0.0, dtype = tf.float32)

    seg_loss = tf.constant(0.0, dtype = tf.float32)
    for k, v in network.get_loss_dict().items():
        # weighting rule below is cited from: https://arxiv.org/abs/1705.07115
        if k.find("seg") == 0:
            seg_loss = seg_loss + v
    seg_s = tf.Variable(0.0, dtype = tf.float32)
    loss = loss + (1.0 * tf.exp(- seg_s) * seg_loss + 0.5 * seg_s)
    
    for weight in network.get_weight_list():
        loss = loss + tf.cast(weight_decay, tf.float32) * 0.5 * tf.reduce_sum(tf.cast(weight, tf.float32) ** 2)
    return loss, [seg_s]

def focal_net(img_h,
              img_w,
              img_ch,
              anchor_size,
              anchor_asp,
              anchor_offset_y,
              anchor_offset_x,
              dtc_words_list,
              seg_words_list):
    
    network = ImageNetwork(img_h, img_w, img_ch, random_seed = None)
    
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(7, 7, 2, 2, True), 32, "relu")
    
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
    
    sub_layer_num = 1
    cls_ch = anchor_size.size * \
             anchor_asp.size * \
             anchor_offset_y.size * \
             anchor_offset_x.size * \
             (1 + len(dtc_words_list))
    reg_ch = anchor_size.size * \
             anchor_asp.size * \
             anchor_offset_y.size * \
             anchor_offset_x.size * \
             4
    cls_weight_list = []
    cls_bias_list = []
    reg_weight_list = []
    reg_bias_list = []
    for l in range(sub_layer_num):
        cls_weight_list.append(network.make_conv_weight(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True),
                                                        input_ch = 256, output_ch = 256))
        cls_bias_list.append(network.make_conv_bias(output_ch = 256))
        reg_weight_list.append(network.make_conv_weight(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True),
                                                        input_ch = 256, output_ch = 256))
        reg_bias_list.append(network.make_conv_bias(output_ch = 256))
    cls_weight = network.make_conv_weight(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True),
                                          input_ch = 256, output_ch = cls_ch)
    cls_bias = network.make_conv_bias(output_ch = cls_ch)
    reg_weight = network.make_conv_weight(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True),
                                       input_ch = 256, output_ch = reg_ch)
    reg_bias = network.make_conv_bias(output_ch = reg_ch)

    # detection
    for i in range(4):
        feature_layer_name = "c{}".format(i + 2)
        # classificatin
        network.add_conv_batchnorm_act(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True), weight = cls_weight_list[0], bias = cls_bias_list[0], activatioin_type = "relu", output_ch = 256, input_name = feature_layer_name)
        for l in range(1, sub_layer_num):
            network.add_conv_batchnorm_act(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True), weight = cls_weight_list[l], bias = cls_bias_list[l], activatioin_type = "relu", output_ch = 256)
        network.add_conv_batchnorm_act(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True), weight = cls_weight, bias = cls_bias, activatioin_type = "relu", output_ch = cls_ch)
        network.add_reshape(shape = [-1,
                                     int(network.get_input(None).get_shape().as_list()[1]),
                                     int(network.get_input(None).get_shape().as_list()[2]),
                                     cls_ch // (1 + len(dtc_words_list)),
                                     1 + len(dtc_words_list)])
        network.add_softmax(name = "cls{}".format(i + 2))
        # regression
        network.add_conv_batchnorm_act(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True), weight = reg_weight_list[0], bias = reg_bias_list[0], activatioin_type = "relu", output_ch = 256, input_name = feature_layer_name)
        for l in range(1, sub_layer_num):
            network.add_conv_batchnorm_act(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True), weight = reg_weight_list[l], bias = reg_bias_list[l], activatioin_type = "relu", output_ch = 256)
        network.add_conv_batchnorm_act(filter_param = ImageNetwork.FilterParam(3, 3, 1, 1, True), weight = reg_weight, bias = reg_bias, activatioin_type = "relu", output_ch = reg_ch)
        network.add_reshape(shape = [-1,
                                     int(network.get_input(None).get_shape().as_list()[1]),
                                     int(network.get_input(None).get_shape().as_list()[2]),
                                     reg_ch // 4,
                                     4])
        network.add_identity(name = "reg{}".format(i + 2))

    # segmentation
    concat_name_list = []
    for i in range(4):
        network.add_upsample(2 ** i, 2 ** i,
                             input_name = "c{}".format(i + 2),
                             name = "seg_ch{}".format(i + 2))
        concat_name_list.append("seg_ch{}".format(i + 2))
    network.add_concat(concat_name_list = concat_name_list)
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(1, 1, 1, 1, True), 256, "relu")
    network.add_conv(ImageNetwork.FilterParam(1, 1, 1, 1, True), 1 + (1 + len(seg_words_list)))
    network.add_softmax(name = "seg")
    
    # detection-loss
    for i in range(4):
        network.add_rect_loss(name = "loss{}".format(i + 2),
                              gamma = 2.0,
                              alpha = 0.5,
                              size_list = anchor_size,
                              asp_list = anchor_asp,
                              offset_y_list = anchor_offset_y,
                              offset_x_list = anchor_offset_x,
                              cls_layer_name = "cls{}".format(i + 2),
                              reg_layer_name = "reg{}".format(i + 2),
                              cls_label_name = "cls_label{}".format(i + 2),
                              reg_label_name = "reg_label{}".format(i + 2))
    
    # segmentation-loss
    network.add_loss(loss_type = "cross_entropy",
                     name = "seg_label",
                     gamma = 2.0)
    
    return network

def evaluate(network, img_h, img_w, 
             anchor_size, anchor_asp, anchor_offset_y, anchor_offset_x, 
             val_type, val_data, tgt_words_list,
             dst_pred_dir, restore_path):
    with tf.Session() as sess:
                    
        if restore_path:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(restore_path)
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                for val_idx in tqdm(range(val_data.get_sample_num(val_type))):
                    # one image
                    img_arr, rect_labels, rects, _1, _2 = val_data.get_vertices_data(val_type, tgt_words_list, index = val_idx, flip = False)
                    eval_feed_dict = make_dtc_feed_dict(network = network,
                                                    img_arr = np.reshape(img_arr, [1] + list(img_arr.shape)),
                                                    rect_labels_list = [rect_labels],
                                                    rects_list = [rects],
                                                    pos_th = 0, neg_th = 0, batch_size = 1)
                    #one_loss = sess.run(total_loss, feed_dict = eval_feed_dict)
                    
                    # output prediction image
                    pal = []
                    pal.append((255,0,0))
                    pal.append((0,255,0))
                    pil_img = Image.fromarray(img_arr.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_img)
                    for i in range(2, 5 + 1):
                        cls, score, rect = decode_anchor_prediction(anchor_cls = sess.run(network.get_layer("cls{}".format(i)), feed_dict = eval_feed_dict),
                                                                    anchor_reg_t = sess.run(network.get_layer("reg{}".format(i)), feed_dict = eval_feed_dict),
                                                                    size_list = anchor_size,
                                                                    asp_list = anchor_asp,
                                                                    offset_y_list = anchor_offset_y,
                                                                    offset_x_list = anchor_offset_x,
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
                    dst_name = "img{0:05d}.png".format(val_idx)
                    # make folder
                    if not os.path.exists(dst_pred_dir):
                        os.makedirs(dst_pred_dir)
                    dst_path = os.path.join(dst_pred_dir, dst_name)
                    pil_img.save(dst_path)
   
def focal_trial():
    
    anchor_size = 4.0 * (2.0 ** (np.arange(0, 2) / 2))
    anchor_asp  = np.linspace(0.5, 2.0, 3)
    anchor_offset_y = np.arange(0, 1) / 1
    anchor_offset_x = np.arange(0, 1) / 1
    img_h  = 256
    img_w  = img_h * 2
    img_ch = 3
    
    dtc_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"]]
    seg_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"],
                      ["road", "drivable area", "ground"]]
    
    #from PIL import Image;Image.fromarray(rgb_arr_).show();exit()
    network = focal_net(img_h  = img_h,
                        img_w  = img_w,
                        img_ch = img_ch,
                        anchor_size = anchor_size,
                        anchor_asp = anchor_asp,
                        anchor_offset_y = anchor_offset_y,
                        anchor_offset_x = anchor_offset_x,
                        dtc_words_list = dtc_words_list,
                        seg_words_list = seg_words_list)
    
    batch_size = 8
    epoch_num = 30
    lr = tf.placeholder(dtype = tf.float32)
    pos_th = 0.5
    neg_th = 0.4
    
#    data = BDD100k(resized_h = img_h,
    data = EdgeAI(resized_h = img_h,
                  resized_w = img_w)
    dtc_loss, dtc_loss_weight_vec = get_dtc_loss(network, weight_decay = 1E-4)
    dtc_opt = tf.train.AdamOptimizer(learning_rate = lr).minimize(dtc_loss)
    seg_loss, seg_loss_weight_vec = get_seg_loss(network, weight_decay = 1E-4)
    seg_opt = tf.train.AdamOptimizer(learning_rate = lr).minimize(seg_loss)
    
    train_type = "train"
    val_type = "val"
    test_type = "test"
    log_interval_sec = 60 * 30
    restore_path = None#r""

    # 軽量化
    pcname = subprocess.getoutput("uname -n")
    if 0:
        if (pcname == "isgsktyktt-VJS111") or \
           (pcname == "Yusuke-PC"):
            train_type = "debug"
            val_type = "debug"
    if (pcname == "isgsktyktt-VJS111"):
        batch_size = 2
            
    if 0:   # ポジティブ判定が出たアンカーを描画
        pal = []
        pal.append((255,0,0))
        pal.append((0,255,0))
        dst_dir = "assigned_anchor"
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        with tf.Session() as sess:
            for i in tqdm(range(data.get_sample_num(train_type))):
                for flip in [False, True]:
                    img_arr, rect_labels, rects, _1, _2 = data.get_vertices_data(train_type, tgt_words_list, index = i, flip = flip)
                    pil_img = Image.fromarray(img_arr.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_img)
                    learn_feed_dict = make_dtc_feed_dict(network,
                                                     1, #batch_size,
                                                     np.reshape(img_arr,     [1] + list(img_arr.shape)),
                                                     np.reshape(rect_labels, [1] + list(rect_labels.shape)),
                                                     np.reshape(rects,       [1] + list(rects.shape)),
                                                     pos_th = pos_th, neg_th = neg_th)
                    # visualize anchored label
                    for l in range(2, 5 + 1):
                        cls = sess.run(network._ImageNetwork__label_dict["cls_label{}".format(l)], feed_dict = learn_feed_dict)
                        reg = make_anchor(network.get_layer("reg{}".format(l)).get_shape().as_list()[1:1+2],
                                          size_list = anchor_size,
                                          asp_list = anchor_asp,
                                          offset_y_list = anchor_offset_y,
                                          offset_x_list = anchor_offset_x)
                        cls = cls.flatten()
                        reg = reg[cls > 0]
                        cls = cls[cls > 0]
                        for j in range(cls.size):
                            draw.rectangle((reg[j][1] * img_w,
                                            reg[j][0] * img_h,
                                            reg[j][3] * img_w,
                                            reg[j][2] * img_h),
                                            outline = pal[cls[j] - 1])
                    pil_img.save(os.path.join(dst_dir, "{0:05d}".format(i) + "{}".format(flip) + ".png"))
        exit()
    
    if 0:   # アンカーが割り当てられたポジティブを描画
        pal = []
        pal.append((255,0,0))
        pal.append((0,255,0))
        dst_dir = "assigned_rect"
        cls_num = 1 + len(tgt_words_list)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        with tf.Session() as sess:
            for i in tqdm(range(data.get_sample_num(train_type))):
                for flip in [False, True]:
                    img_arr, rect_labels, rects, _1, _2 = data.get_vertices_data(train_type, tgt_words_list, index = i, flip = flip)
                    pil_img = Image.fromarray(img_arr.astype(np.uint8))
                    draw = ImageDraw.Draw(pil_img)
                    if (rect_labels.size > 0) and (rects.size > 0):
                        learn_feed_dict = make_dtc_feed_dict(network,
                                                         1, #batch_size,
                                                         np.reshape(img_arr,     [1] + list(img_arr.shape)),
                                                         np.reshape(rect_labels, [1] + list(rect_labels.shape)),
                                                         np.reshape(rects,       [1] + list(rects.shape)),
                                                         pos_th = pos_th, neg_th = neg_th)
                        # visualize anchored label
                        for l in range(2, 5 + 1):
                            cls, score, rect = decode_anchor_prediction(anchor_cls = sess.run(tf.one_hot(network._ImageNetwork__label_dict["cls_label{}".format(l)], depth = cls_num), feed_dict = learn_feed_dict),
                                                                        anchor_reg_t = sess.run(network._ImageNetwork__label_dict["reg_label{}".format(l)], feed_dict = learn_feed_dict),
                                                                        size_list = anchor_size,
                                                                        asp_list = anchor_asp,
                                                                        offset_y_list = anchor_offset_y,
                                                                        offset_x_list = anchor_offset_x,
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
                    dst_path = os.path.join(dst_dir, "{0:05d}".format(i) + "{}".format(flip) + ".png")
                    pil_img.save(dst_path)
        exit()
    
    if 0:	# evaluate-only
        dst_pred_dir = r"C:\Users\Yusuke\workspace\tmp_out"
        restore_path = r"C:\Users\Yusuke\workspace\deeplearning\result_20181123_221648\model\epoch0000_batch6288"
        if not os.path.exists(dst_pred_dir):
            os.makedirs(dst_pred_dir)
        evaluate(network, img_h, img_w, anchor_size, anchor_asp, anchor_offset_y, anchor_offset_x, test_type,
         data, tgt_words_list, 
         dst_pred_dir,
         restore_path)
        exit()

    result_dir = "result_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(result_dir)
    with tf.Session() as sess:
        tf.summary.FileWriter(os.path.join(result_dir, "graph"), sess.graph)
        saver = tf.train.Saver()

        # restore
        if restore_path:
            ckpt = tf.train.get_checkpoint_state(restore_path)
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        
        start_time = time.time()
        tmp_out = False
        def save_model(epoch, b):
            dst_model_dir = os.path.join(result_dir, "model", "epoch{0:04d}".format(epoch) + "_batch{}".format(b))
            if not os.path.exists(dst_model_dir):
                os.makedirs(dst_model_dir)
            dst_model_path = os.path.join(dst_model_dir, "model.ckpt")
            saver.save(sess, dst_model_path)
            return dst_model_dir
        
        for epoch in range(epoch_num):
            for b in tqdm(range(data.get_sample_num(train_type) // batch_size)):
                
                # one image
                rect_labels_list = []
                rects_list = []
                img_arr_list = np.empty((batch_size, img_h, img_w, img_ch)).astype(np.int)
                seg_arr_list = []
                learn_feed_dict = {lr : 1e-2}
                if b % 2 == 0:
                    for batch_cnt in range(batch_size):
                        img_arr, rect_labels, rects, _1, _2 = data.get_vertices_data(train_type, dtc_words_list, index = None, flip = None)
                        img_arr_list[batch_cnt] = img_arr
                        rect_labels_list.append(rect_labels)
                        rects_list.append(rects)
                    learn_feed_dict.update(make_dtc_feed_dict(network, batch_size, img_arr_list, rect_labels_list, rects_list, pos_th = pos_th, neg_th = neg_th))
                    print(epoch, b, sess.run([dtc_opt, dtc_loss, dtc_loss_weight_vec], feed_dict = learn_feed_dict))
                else:
                    for batch_cnt in range(batch_size):
                        img_arr, seg_arr = data.get_seg_data(train_type, dtc_words_list, index = None, flip = None)
                        img_arr_list[batch_cnt] = img_arr
                        seg_arr_list.append(seg_arr)
                    learn_feed_dict.update(make_seg_feed_dict(network, batch_size, img_arr_list, np.array(seg_arr_list)))
                    print(epoch, b, sess.run([seg_opt, seg_loss, seg_loss_weight_vec], feed_dict = learn_feed_dict))
                
                if (time.time() - start_time >= log_interval_sec) or ((epoch == epoch_num - 1)and(b == data.get_sample_num(train_type) // batch_size - 1)):
                    # Save model
                    save_model(epoch, b)
                    start_time = time.time()
                
                dst_pred_dir = os.path.join(result_dir, "tmp_out")
                if 1:
                    check = os.path.exists(dst_pred_dir)
                    if check:
                        if os.path.isdir(dst_pred_dir):
                            if tmp_out is False:
                                evaluate(network, img_h, img_w, anchor_size, anchor_asp, anchor_offset_y, anchor_offset_x, train_type,
                                         data, dtc_words_list, dst_pred_dir,
                                         restore_path = save_model(epoch, b))
                                tmp_out = True
                    else:
                        tmp_out = False
if "__main__" == __name__:
    focal_trial()