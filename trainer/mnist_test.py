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
    def __init__(self, model, x, y_dict, lr = 1e-3, opt_type = "adam"):
        self.__model = model
        self.__label_dict = {}
        total_loss = 0.0
        for k, v in y_dict.items():
            total_loss = total_loss + self.__model.get_loss()[k]
            self.__label_dict[k] = y_dict[k]
        if opt_type == "adam":
            self.__optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss)
        else:
            assert(0)
        self.__x = x
        self.__y_dict = y_dict
        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())

    def training(self, lr, batch_size, valid_loss_dict = None):
        label_dict = {}

        batch_idx = np.random.choice(np.arange(self.__x.shape[0]), batch_size, replace = True)
        batch_x = self.__x[batch_idx]
    
        #answer_layer = network.get_layer(name = "answer")
        #shape_list = [None] + list(self.__y.shape[1:])
        #y = tf.placeholder(dtype = tf.float32,
        #                   shape = shape_list)
        
        #correct = tf.equal(tf.argmax(y, axis = 1), tf.argmax(answer_val, axis = 1))
        #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        label_dict = {}
        for k, v in self.__label_dict.items():
            label_dict[k] = v[batch_idx]
        feed_dict = self.__model.make_feed_dict(input_image = batch_x,
                                                label_dict = label_dict,
                                                is_training = True)
        self.__sess.run(self.__optimizer, feed_dict = feed_dict)
    
    def predict(self, x, eval_name_list):
        out = []
        feed_dict = self.__model.make_feed_dict(input_image = x,
                                                is_training = False)
        for eval_name in eval_name_list:
            out.append(self.__sess.run(self.__model.get_layer(eval_name), feed_dict = feed_dict))
        return out    
if __name__ == "__main__":
    
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
    network.add_dropout(0.5)
    network.add_full_connect(10)
    network.add_softmax("answer")
    network.add_loss("cross_entropy", name = "ce_loss")
    #network.show()
    
    batch_size = 8
    epoch_num = 10
    lr = 1e-6
    x, y = mnist.get_data("train")
    y = transform_one_hot(y, 10)
    xv, yv = mnist.get_data("val")
    trainer = Trainer(model = network,
                      x = x,
                      y_dict = {"ce_loss":y})
    
    for epoch in range(epoch_num):
        for b in (range(x.shape[0] // batch_size)):
            trainer.training(lr = lr,
                             batch_size = batch_size,
                             valid_loss_dict = None)
            ans_mat = trainer.predict(xv, ["answer"])[0]
            ans = np.argmax(ans_mat, axis = 1)
            print(np.average(ans == yv))
