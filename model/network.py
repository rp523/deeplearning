#coding: utf-8
import os, sys, shutil
import numpy as np
import tensorflow as tf


class ImageNetwork:
    def __init__(self, image_h, image_w, image_ch,
                 dtype = None,
                 random_seed = None):
        tf.reset_default_graph()    # reset old settings
        src_img_layer = tf.placeholder(dtype = tf.float32,
                                       shape = [None, image_h, image_w, image_ch],
                                       name = "input_image")
        self.__layer_list = [src_img_layer]
        
        # default type of variables
        if dtype is not None:
            self.__default_type = dtype
        else:
            self.__default_type = tf.float32
        
        # for dropout
        self.__dropout_ph_list = []
        self.__dropout_val_list = []
        
        # set random seed
        if random_seed is None:
            random_seed = 0
        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)
        
        # loss, label
        self.__loss_dict = {}
        self.__label_dict = {}
        
    def __get_padding_str(self, padding):
        assert(isinstance(padding, bool))
        if padding:
            return "SAME"
        else:
            return "VALID"
    
    def __get_dtype(self, dtype):
        if dtype is None:
            return self.__default_type
        else:
            return dtype
    
    def __get_layer_by_name(self, name):
        assert(name is not None)
        ret_layer = None
        for layer in self.__layer_list:
            if layer.name[:layer.name.rfind(":")] == name:
                ret_layer = layer
                break
        return ret_layer
    
    def add_layer(self, new_layer):
        name = new_layer.name
        if name is not None:
            assert(self.__get_layer_by_name(name) is None)
        self.__layer_list.append(new_layer)
    
    def get_input(self, name):
        if name is not None:
            return self.__get_layer_by_name(name)
        else:
            return self.__layer_list[-1]
    
    def show(self):
        for layer in self.__layer_list:
            print(layer.name, layer.get_shape())
        
    def add_full_connect(self, output_ch, name = None, input_name = None, dtype = None):
        input_layer = self.get_input(input_name)
        input_ch = int(np.prod(input_layer.get_shape()[1:]))
        input_layer = tf.reshape(input_layer, [-1, input_ch])   # flatten
        
        weight = tf.get_variable("fc_weight{}".format(len(self.__layer_list)),
                                 shape = [input_ch, output_ch],
                                 initializer = tf.contrib.layers.xavier_initializer(),
                                 dtype = self.__get_dtype(dtype))
        bias = tf.get_variable("fc_bias{}".format(len(self.__layer_list)),
                               shape = [output_ch],
                               initializer = tf.contrib.layers.xavier_initializer(),
                               dtype = self.__get_dtype(dtype))
        
        new_layer = tf.add(tf.matmul(input_layer, weight), bias, name = name)
        self.add_layer(new_layer)
    
    class FilterParam:
        def __init__(self, kernel_y, kernel_x, stride_y, stride_x, padding):
            self.kernel_y = kernel_y
            self.kernel_x = kernel_x
            self.stride_y = stride_y
            self.stride_x = stride_x
            self.padding  = padding

    def add_conv(self, filter_param, output_ch, bias = True, name = None, input_name = None, dtype = None):
        input_layer = self.get_input(input_name)
        new_layer = self.make_conv(input_layer, filter_param, output_ch, bias, name)
        self.add_layer(new_layer)
        
    def make_conv(self, input_layer, filter_param, output_ch, bias = True, name = None, dtype = None):
        assert(len(input_layer.get_shape()) == 4)
    
        input_ch = int(input_layer.get_shape()[-1])
        
        filter = tf.get_variable("conv_filter{}".format(len(self.__layer_list)),
                                  shape = [filter_param.kernel_y, filter_param.kernel_x, input_ch, output_ch],
                                  initializer = tf.contrib.layers.xavier_initializer(),
                                  dtype = self.__get_dtype(dtype))
        new_layer = tf.nn.conv2d(input = input_layer,
                                 filter = filter,
                                 strides=[1, filter_param.stride_y, filter_param.stride_x, 1],
                                 padding = self.__get_padding_str(filter_param.padding))
        
        if bias is True:
            bias = tf.get_variable("conv_bias{}".format(len(self.__layer_list)),
                                   shape = [output_ch],
                                   initializer = tf.contrib.layers.xavier_initializer(),
                                   dtype = self.__get_dtype(dtype))
            new_layer = tf.add(new_layer, bias)
        return new_layer

    def add_pool(self, pool_type, filter_param, name = None, input_name = None):
        input_layer = self.get_input(input_name)
        assert(len(input_layer.get_shape()) == 4)
        if pool_type == "MAX":
            pool_func = tf.nn.max_pool
        elif pool_type == "AVG":
            pool_func = tf.nn.avg_pool
        else:
            assert(0)
        new_layer = pool_func(value   = input_layer,
                              ksize   = [1, filter_param.kernel_y, filter_param.kernel_x, 1],
                              strides = [1, filter_param.stride_y, filter_param.stride_x, 1],
                              padding = self.__get_padding_str(filter_param.padding),
                              name = name)
        self.add_layer(new_layer)

    def add_dropout(self, prob_val, name = None, input_name = None, dtype = None):
        input_layer = self.get_input(input_name)
        keep_prob = tf.placeholder(dtype = self.__get_dtype(dtype))
        new_layer = tf.nn.dropout(x = input_layer,
                                  keep_prob = keep_prob, 
                                  name = name)
        self.__dropout_ph_list.append(keep_prob)
        self.__dropout_val_list.append(prob_val)
        self.add_layer(new_layer)
    
    def add_activation(self, activatioin_type, name = None, input_name = None):
        input_layer = self.get_input(input_name)
        if activatioin_type == "relu":
            activ_func = tf.nn.relu
        elif activatioin_type == "tanh":
            activ_func = tf.nn.tanh
        else:
            assert(0)
        new_layer = activ_func(input_layer)
        self.add_layer(new_layer)
    
    def add_batchnorm(self, global_norm = True, name = None, input_name = None, dtype = None):
        input_layer = self.get_input(input_name)
        input_ch = int(input_layer.get_shape()[-1])
        if global_norm is True:
            axes = [0, 1, 2]
        else:
            axes = [0]
        mean, variance = tf.nn.moments(x = input_layer,
                                       axes = axes)
        scale  = tf.Variable(tf.ones([input_ch]),
                             dtype = self.__get_dtype(dtype))
        offset = tf.Variable(tf.zeros([input_ch]),
                             dtype = self.__get_dtype(dtype))
        
        new_layer = tf.nn.batch_normalization(x = input_layer,
                                              mean = mean,
                                              variance = variance,
                                              offset = offset,
                                              scale = scale,
                                              variance_epsilon = 1e-4,
                                              name = name)
        self.add_layer(new_layer)
    
    def add_conv_batchnorm(self, filter_param, output_ch, bias = True, global_norm = True, name = None, input_name = None):
        self.add_conv(filter_param, output_ch, bias, input_name = input_name)
        self.add_batchnorm(global_norm, name)
        
    def add_softmax(self, name = None, input_name = None):
        input_layer = self.get_input(input_name)
        new_layer = tf.nn.softmax(logits = input_layer,
                                  name = name)
        self.add_layer(new_layer)
    
    def add_branching(self, name, input_name = None):
        input_layer = self.get_input(input_name)
        new_layer = tf.identity(input = input_layer, name = name)
        self.add_layer(new_layer)
    
    def add_injection(self, injection_name, input_name = None, name = None):
        input_layer = self.get_input(input_name)
        injection_layer = self.get_input(injection_name)
        assert(injection_layer is not None)
        new_layer = tf.add(x = input_layer,
                           y = injection_layer,
                           name = name)
        self.add_layer(new_layer)
    
    def add_loss(self, loss_type, input_name, name):
        answer_layer = self.get_input(input_name)
        label = tf.placeholder(dtype = tf.float32,
                               shape = [None] + answer_layer.get_shape().tolist())
        if loss_type == "cross_entropy":
            loss = - tf.reduce_mean(label * tf.log(answer_layer + 1e-5))
        else:
            assert(0)
        assert(not name in self.__loss_dict.keys())
        assert(not name in self.__label_dict.keys())
        self.__loss_dict[name]  = loss
        self.__label_dict[name] = label

    def get_loss(self, name_list, weight_dict = None):
        assert(len(name_list) == len(weight_dict.keys()))
        total_loss = 0.0
        for name in name_list:
            assert(name in self.__loss_dict.keys())
            if weight_dict is not None:
                assert(name in weight_dict.keys())
                weight = weight_dict[name]
            else:
                weight = 1.0
            total_loss = total_loss + weight * self.__loss_dict[name]
        return total_loss
        
    '''        
    def __get_optimizer(self, loss, optimizer_type, learning_rate):
        if optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else:
            assert(0)
        return optimizer
    '''
    def make_feed_dict(self, input_image, label_dict, is_training):
        feed_dict = {}

        # input image
        feed_dict[self.__layer_list[0]] = input_image
        
        for name, label in label_dict:
            feed_dict[self.__label_dict[name]] = label
        
        # for dropout
        if is_training is True:
            for dropout_idx in range(len(self.__dropout_ph_list)):
                feed_dict[self.__dropout_ph_list[dropout_idx]] = self.__dropout_val_list[dropout_idx]
        else:
            for dropout_idx in range(len(self.__dropout_ph_list)):
                feed_dict[self.__dropout_ph_list[dropout_idx]] = 1.0
        
        return feed_dict
    
    # to be deleted
    def fit(self, train_x, train_y, valid_x, valid_y, loss_type, optimizer_type, learning_rate, epoch_num, batch_size, log_interval = 1):
        assert(train_x.shape[1:] == valid_x.shape[1:])
        assert(train_y.shape[1:] == valid_y.shape[1:])
        assert(train_x.shape[0] == train_y.shape[0])
        assert(valid_x.shape[0] == valid_y.shape[0])
        shape_list = [None]
        for dim in train_y.shape[1:]:
            shape_list.append(dim)
        y = tf.placeholder(dtype = tf.float32,
                           shape = shape_list)
        loss = self.__get_loss(loss_type, y)
        optimizer = self.__get_optimizer(loss, optimizer_type, learning_rate)
        
        correct = tf.equal(tf.argmax(y, axis = 1), tf.argmax(self.__layer_list[-1], axis = 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(epoch_num):
                batches = self.__make_batches(sample_num = train_x.shape[0],
                                              batch_size = batch_size)
                batch_cnt = 0
                for train_idx in range(len(batches)):
                    batch = batches[train_idx]
                    batch_x = train_x[batch]
                    batch_y = train_y[batch]
    
                    train_feed = self.__make_feed_dict(y, batch_x, batch_y, True)
                    sess.run(optimizer, feed_dict = train_feed)
                    
                    batch_cnt = batch_cnt + batch.size
                    valid_feed = self.__make_feed_dict(y, valid_x, valid_y, False)
                    if log_interval > 0:
                        if train_idx % log_interval == 0:
                            acc = sess.run(accuracy, feed_dict = valid_feed)
                            print("[epoch:{epoch}/{epoch_tot}][batch:{batch}/{batch_tot}]{acc}".format(
                                epoch = epoch,
                                epoch_tot = epoch_num,
                                batch = batch_cnt,
                                batch_tot = train_x.shape[0],
                                acc = acc))

def main():
    from mnist import Mnist
    mnist = Mnist()
    train_x = mnist.get_train_images().reshape(-1, 28, 28, 1) / 255 * 2 - 1
    valid_x = mnist.get_test_images().reshape(-1, 28, 28, 1) / 255 * 2 - 1
    train_y = mnist.get_train_labels(one_hot = True)
    valid_y = mnist.get_test_labels(one_hot = True)
    
    network = Network(image_h = 28,
                      image_w = 28,
                      image_ch = 1)
    network.add_conv(Network.FilterParam(3, 3, 1, 1, True), 32)
    network.add_batchnorm()
    network.add_activation("relu")
    network.add_pool("MAX", Network.FilterParam(2, 2, 2, 2, True))
    #network.add_coordconv(Network.FilterParam(3, 3, 1, 1, True), 64)
    #network.add_batchnorm()
    #network.add_activation("relu")
    #network.add_pool("MAX", Network.FilterParam(2, 2, 2, 2, True))
    network.add_full_connect(1024)
    network.add_dropout(0.5)
    network.add_full_connect(10)
    network.add_softmax()
    network.show()
    
    network.fit(train_x = train_x,
                train_y = train_y,
                valid_x = valid_x,
                valid_y = valid_y,
                loss_type = "cross_entropy",
                optimizer_type = "adam",
                learning_rate = 1e-4,
                epoch_num = 10,
                batch_size = 32,
                log_interval = 10)
    
if "__main__" == __name__:
    main()
    print("Done.")