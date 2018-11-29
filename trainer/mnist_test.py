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
    N = 7
    network = ImageNetwork(image_h = 28,
                           image_w = 28,
                           image_ch = 1,
                           input_dtype = tf.float32,
                           dtype = tf.float32)
    for i, filter_param in enumerate([ImageNetwork.FilterParam(3, 3, 1, 1, True),
                                      ImageNetwork.FilterParam(3, 3, 2, 2, True),
                                      ImageNetwork.FilterParam(3, 3, 2, 2, True)]):
        output_ch = 2 ** (i + 4)
        input_ch = network.get_input(None).get_shape().as_list()[3]
        weight = network.make_conv_weight(filter_param, input_ch = input_ch, output_ch = output_ch)
        bias   = network.make_conv_bias(output_ch = output_ch)
        conv = network.make_conv(filter_param, output_ch, weight = weight, bias = bias)
        q_conv = network.make_quantize(N, conv)
        network.add_switch(conv, q_conv)
        
        #network.add_batchnorm()
        network.add_groupnorm(group_div = 16)
        network.add_activation("relu")
        layer = network.get_input(None)
        q_layer = network.make_quantize(N, layer)
        network.add_switch(layer, q_layer)
    
    network.add_full_connect(1024)
    layer = network.get_input(None)
    q_layer = network.make_quantize(N, layer)
    network.add_switch(layer, q_layer)

    network.add_activation("relu")
    layer = network.get_input(None)
    q_layer = network.make_quantize(N, layer)
    network.add_switch(layer, q_layer)

    network.add_dropout(0.5)
    
    network.add_full_connect(10)
    layer = network.get_input(None)
    q_layer = network.make_quantize(N, layer)
    network.add_switch(layer, q_layer)
    
    network.add_softmax("answer")
    network.add_loss("cross_entropy", name = "ce_loss")
    #network.show()
    
    total_loss = tf.constant(0.0)
    for loss in network.get_loss_dict().values():
        total_loss += loss
    batch_size = 8
    epoch_num = 100
    lr = 1e-4
    x, y = mnist.get_data("train")
    y = transform_one_hot(y, 10)
    xv, yv = mnist.get_data("val")
    yv = transform_one_hot(yv, 10)
    optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            batch_cnt = 0
            for b in (range(x.shape[0] // batch_size)):
                batch_cnt += batch_size
                batch_idx = np.random.choice(np.arange(x.shape[0]), batch_size, replace = True)
                learn_feed_dict = network.create_feed_dict(input_image = x[batch_idx] / 128 - 0.5, is_training = True, label_dict = {"ce_loss": y[batch_idx]})
                sess.run(optimizer, feed_dict = learn_feed_dict)
                if b % 10 == 0:
                    eval_feed_dict = network.create_feed_dict(input_image = xv / 128 - 0.5, is_training = False, label_dict = {"ce_loss": yv})
                    eval_ans_mat = sess.run(network.get_layer("answer"), feed_dict = eval_feed_dict)
                    eval_acc = np.average(np.argmax(eval_ans_mat, axis = 1) == np.argmax(yv, axis = 1))
                    learn_feed_dict = network.create_feed_dict(input_image = xv / 128 - 0.5, is_training = True, label_dict = {"ce_loss": yv})
                    learn_ans_mat = sess.run(network.get_layer("answer"), feed_dict = learn_feed_dict)
                    learn_acc = np.average(np.argmax(learn_ans_mat, axis = 1) == np.argmax(yv, axis = 1))
                    print("[epoch={e}/{et}][batch={b}/{bt}] acc={lacc},{eacc}".format(e = epoch, et = epoch_num, b = batch_cnt, bt = x.shape[0],
                                                                              lacc = learn_acc,
                                                                              eacc = eval_acc))

if "__main__" == __name__:
    mnist_trial()