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
                           image_ch = 1,
                           input_dtype = tf.float16,
                           dtype = tf.float16)
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 32, "ste")
    network.add_pool("MAX", ImageNetwork.FilterParam(2, 2, 2, 2, True))
    network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, 1, 1, True), 64, "ste")
    network.add_pool("MAX", ImageNetwork.FilterParam(2, 2, 2, 2, True))
    network.add_full_connect(1024)
    network.add_activation("ste")
    #network.add_dropout(0.5)
    network.add_full_connect(10)
    network.add_softmax("answer")
    network.add_loss("cross_entropy", name = "ce_loss")
    #network.show()
    
    total_loss = network.get_total_loss(weight_decay = 1E-4)
    batch_size = 32
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
                    ans_mat = sess.run(network.get_layer("answer"), feed_dict = eval_feed_dict)
                    acc = np.average(np.argmax(ans_mat, axis = 1) == np.argmax(yv, axis = 1))
                    print("[epoch={e}/{et}][batch={b}/{bt}] acc={acc}".format(e = epoch, et = epoch_num, b = batch_cnt, bt = x.shape[0], acc = acc))

if "__main__" == __name__:
    mnist_trial()