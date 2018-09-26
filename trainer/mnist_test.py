#coding: utf-8
import os, sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../")
from dataset.mnist import MNIST
from model.network import ImageNetwork
from transformer import *

class Trainer:
    def __init__(self, model, x, y_dict, opt_type = "adam"):
        self.__model = model
        if opt_type == "adam":
            self.__opt_func = tf.train.AdamOptimizer
        else:
            assert(0)
        self.__x = x
        self.__y_dict = y_dict
        #self.__sess = tf.Session()
        #self.__sess.run(tf.global_variables_initializer())

    def training(self, lr, batch_size, tgt_loss_list, valid_loss_dict = None):
        total_loss = 0.0
        label_dict = {}

        batch_idx = np.random.choice(np.arange(self.__x.shape[0]), batch_size, replace = True)
        batch_x = self.__x[batch_idx]
        for tgt_loss in tgt_loss_list:
            total_loss = total_loss + self.__model.get_loss()[tgt_loss]
            label_dict[tgt_loss] = self.__y_dict[tgt_loss][batch_idx]
        optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss)
    
        #answer_layer = network.get_layer(name = "answer")
        #shape_list = [None] + list(self.__y.shape[1:])
        #y = tf.placeholder(dtype = tf.float32,
        #                   shape = shape_list)
        
        #correct = tf.equal(tf.argmax(y, axis = 1), tf.argmax(answer_val, axis = 1))
        #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        feed_dict = self.__model.make_feed_dict(input_image = batch_x,
                                                label_dict = label_dict,
                                                is_training = True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(optimizer, feed_dict = feed_dict)
    
        
if __name__ == "__main__":
    
    mnist = MNIST()

    network = ImageNetwork(image_h = 28,
                           image_w = 28,
                           image_ch = 1)
    network.add_conv(ImageNetwork.FilterParam(3, 3, 1, 1, True), 32)
    network.add_batchnorm()
    network.add_activation("relu")
    network.add_pool("MAX", ImageNetwork.FilterParam(2, 2, 2, 2, True))
    #network.add_coordconv(Network.FilterParam(3, 3, 1, 1, True), 64)
    #network.add_batchnorm()
    #network.add_activation("relu")
    #network.add_pool("MAX", Network.FilterParam(2, 2, 2, 2, True))
    network.add_full_connect(1024)
    network.add_dropout(0.5)
    network.add_full_connect(10)
    network.add_softmax("answer")
    network.add_loss("cross_entropy", name = "ce_loss")
    #network.show()
    
    batch_size = 32
    epoch_num = 10
    lr = 1e-3
    x, y = mnist.get_data("train")
    y = transform_one_hot(y, 10)
    trainer = Trainer(model = network,
                      x = x,
                      y_dict = {"ce_loss":y})
    for epoch in range(epoch_num):
        print(epoch)
        for b in tqdm(range(x.shape[0] // batch_size)):
            print("b=",b, end="")
            trainer.training(lr = lr,
                             batch_size = batch_size,
                             tgt_loss_list = ["ce_loss"],
                             valid_loss_dict = None)
            print(b)