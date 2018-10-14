#coding; utf-8
import os, sys
import numpy as np
import tensorflow as tf

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
