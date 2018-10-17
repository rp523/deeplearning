#coding; utf-8
import os, sys
import numpy as np
import tensorflow as tf

class Trainer:
    def __init__(self, model, total_loss, lr = 1e-3, opt_type = "adam"):
        self.__model = model
        if opt_type == "adam":
            self.__optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss)
        else:
            assert(0)
        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())

    def training(self, feed_dict):
        self.__sess.run(self.__optimizer, feed_dict = feed_dict)
    
    def predict(self, x, eval_name_list):
        out = []
        feed_dict = self.__model.make_feed_dict(input_image = x,
                                                is_training = False)
        for eval_name in eval_name_list:
            out.append(self.__sess.run(self.__model.get_layer(eval_name), feed_dict = feed_dict))
        return out    
