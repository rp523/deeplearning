#coding: utf-8
import os, sys, shutil
import numpy as np
import tensorflow as tf
from transformer import *
import time

class ImageNetwork:
    def __init__(self, image_h, image_w, image_ch,
                 input_dtype = None,
                 dtype = None,
                 random_seed = None):
        tf.reset_default_graph()    # reset old settings
        
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
        
        self.__weight_list = []
        self.__bias_list = []
        self.__anchor = {}
        self.__anchor_ph = {}
        
        # only for debug
        self.debug = []
        
        if input_dtype is not None:
            input_dtype = input_dtype
        else:
            input_dtype = tf.float32
        src_img_layer = tf.placeholder(dtype = input_dtype,
                                       shape = [None, image_h, image_w, image_ch],
                                       name = "input_image")
        self.__layer_list = [src_img_layer]
        self.__is_training = tf.placeholder(dtype = tf.bool)
        self.__saved_time = None
        
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
    
    def get_layer(self, name):
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
            assert(self.get_layer(name) is None)
        self.__layer_list.append(new_layer)
    
    def get_input(self, name):
        if name is not None:
            return self.get_layer(name)
        else:
            return self.__layer_list[-1]
    
    def show_all(self):
        for layer in self.__layer_list:
            print(layer.name, layer.get_shape())

    def show_layer(self, name):
        layer = self.get_layer(name)
        print(layer.name, layer.get_shape())
        
    def add_reshape(self, shape, name = None, input_name = None, dtype = None):
        input_layer = self.get_input(input_name)
        new_layer = tf.reshape(tensor = input_layer,
                               shape = shape,
                               name = name)
        self.add_layer(new_layer)
        
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
        self.__weight_list.append(weight)
        
    def add_upsample(self, scale_y, scale_x, upType = "nn", name = None, input_name = None):
        input_layer = self.get_input(input_name)
        input_shape = input_layer.get_shape().as_list()
        if upType == "nn":
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif upType == "bilinear":
            method = tf.image.ResizeMethod.BILINEAR
        elif upType == "bicubic":
            method = tf.image.ResizeMethod.BICUBIC
        elif upType == "area":
            method = tf.image.ResizeMethod.AREA
        else:
            assert(0 & upType)
        new_layer = tf.image.resize_images(
                        input_layer,
                        [input_shape[1] * scale_y, input_shape[2] * scale_x],
                        method = method)
        if name is not None:
            new_layer = tf.identity(new_layer, name = name)
        self.add_layer(new_layer)
    
    class FilterParam:
        def __init__(self, kernel_y, kernel_x, stride_y, stride_x, padding):
            self.kernel_y = kernel_y
            self.kernel_x = kernel_x
            self.stride_y = stride_y
            self.stride_x = stride_x
            self.padding  = padding

    def make_conv_weight(self, filter_param, input_ch, output_ch, dtype = None):
        weight = tf.get_variable("conv_filter{}".format(len(self.__weight_list)),
                                  shape = [filter_param.kernel_y, filter_param.kernel_x, input_ch, output_ch],
                                  initializer = tf.contrib.layers.xavier_initializer(),
                                  dtype = self.__get_dtype(dtype))
        self.__weight_list.append(weight)
        return weight
    
    def make_conv_bias(self, output_ch, dtype = None):
        bias = tf.get_variable("conv_bias{}".format(len(self.__bias_list)),
                               shape = [output_ch],
                               initializer = tf.contrib.layers.xavier_initializer(),
                               dtype = self.__get_dtype(dtype))
        self.__bias_list.append(bias)
        return bias
    
    def make_conv(self, filter_param, output_ch, name = None, input_name = None, dtype = None, weight = None, bias = None):
        input_layer = self.get_input(input_name)
        assert(input_layer is not None)
        assert(len(input_layer.get_shape()) == 4)
    
        input_ch = int(input_layer.get_shape()[-1])
        
        if weight is None:
            weight = self.make_conv_weight(filter_param, input_ch, output_ch, dtype)
        if bias is None:
            bias   = self.make_conv_bias(output_ch, dtype)
        new_layer = tf.nn.conv2d(input = input_layer,
                                 filter = weight,
                                 strides=[1, filter_param.stride_y, filter_param.stride_x, 1],
                                 padding = self.__get_padding_str(filter_param.padding))
        new_layer = new_layer + bias
        new_layer = tf.identity(new_layer, name = name)
        return new_layer
    
    def add_conv(self, filter_param, output_ch, name = None, input_name = None, dtype = None, weight = None, bias = None):
        new_layer = self.make_conv(filter_param, output_ch, name, input_name, dtype, weight, bias)
        self.add_layer(new_layer)

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
            new_layer = tf.nn.relu(input_layer, name = name)
        elif activatioin_type == "tanh":
            new_layer = tf.nn.tanh(input_layer, name = name)
        elif activatioin_type == "ste":
            new_layer = tf.cond(self.__is_training,
                                lambda:tf.clip_by_value(input_layer, - 1.0, 1.0),
                                lambda:tf.sign(input_layer))
        else:
            assert(0)
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
        scale  = tf.Variable(tf.ones([input_ch], dtype = self.__get_dtype(dtype)),
                             dtype = self.__get_dtype(dtype))
        offset = tf.Variable(tf.zeros([input_ch], dtype = self.__get_dtype(dtype)),
                             dtype = self.__get_dtype(dtype))
        
        new_layer = tf.nn.batch_normalization(x = input_layer,
                                              mean = mean,
                                              variance = variance,
                                              offset = offset,
                                              scale = scale,
                                              variance_epsilon = 1e-4,
                                              name = name)
        self.add_layer(new_layer)
    
    def add_groupnorm(self, group_div, name = None, input_name = None, dtype = None):
        input_layer = self.get_input(input_name)
        N, H, W, C = input_layer.get_shape()
        assert(C % group_div == 0)
        new_layer = tf.reshape(input_layer, [-1, int(H), int(W), int(C // group_div), int(group_div)])
        mean, var = tf.nn.moments(new_layer, axes = [1, 2, 3], keep_dims = True)
        new_layer = (new_layer - mean) / tf.sqrt(var + 1e-5)
        new_layer = tf.reshape(new_layer, [-1, H, W, C])
        self.add_layer(new_layer)
        
    def add_conv_act(self, filter_param, output_ch, activatioin_type, bias = True, name = None, input_name = None):
        self.add_conv(filter_param, output_ch, bias, input_name = input_name)
        self.add_activation(activatioin_type, name)
        
    def add_conv_batchnorm(self, filter_param, output_ch, bias = True, global_norm = True, name = None, input_name = None):
        self.add_conv(filter_param, output_ch, bias, input_name = input_name)
        self.add_batchnorm(global_norm, name)

    def add_conv_batchnorm_act(self, filter_param, output_ch, activatioin_type, global_norm = True, name = None, input_name = None, weight = None, bias = None):
        self.add_conv(filter_param = filter_param, output_ch = output_ch, input_name = input_name, weight = weight, bias = bias)
        self.add_batchnorm(global_norm)
        self.add_activation(activatioin_type, name)

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
    
    def add_identity(self, name, input_name = None):
        input_layer = self.get_input(input_name)
        new_layer = tf.identity(input = input_layer, name = name)
        self.add_layer(new_layer)

    def add_concat(self, concat_name_list, name = None):
        concat_layer_list = []
        for concat_name in concat_name_list:
            concat_layer = self.get_input(concat_name)
            assert(None != concat_layer)
            concat_layer_list.append(concat_layer)
        new_layer = tf.concat(concat_layer_list,
                              axis = 3,
                              name = name)
        self.add_layer(new_layer)

    def get_loss_dict(self):
        return self.__loss_dict
    
    def get_weight_list(self):
        return self.__weight_list

    def add_loss(self, loss_type, name, input_name = None, gamma = 1.0, masking = False):
        pred_layer = self.get_input(input_name)
        # lossはすべてfloat32
        label = tf.placeholder(dtype = tf.float32,
                               shape = [None] + list(pred_layer.get_shape())[1:])
        if masking:
            mask = tf.placeholder(dtype = tf.bool,
                                  shape = [None] + list(pred_layer.get_shape())[1:])
        if loss_type == "cross_entropy":
            p_t = (1.0 - pred_layer) + (2.0 * pred_layer - 1.0) * label
            loss_vec = - ((1.0 - p_t) ** gamma) * tf.log(p_t + 1e-5)
        elif loss_type == "L1":
            loss_vec = label - pred_layer
        else:
            assert(0)
        if masking:
            loss_vec = tf.boolean_mask(loss_vec, mask)
        loss = tf.reduce_mean(loss_vec)
        assert(not name in self.__loss_dict.keys())
        self.__loss_dict[name]  = loss
        assert(not name in self.__label_dict.keys())
        self.__label_dict[name] = label
        if masking:
            assert(not name + "_mask" in self.__label_dict.keys())
            self.__label_dict[name + "_mask"] = mask

    def make_quantize(self, N, input_layer):
        sign = tf.sign(input_layer)
        val = tf.abs(input_layer)
        y = tf.log(val) / tf.log(2.0) + N
        y = tf.maximum(y, 0)
        y = tf.cast(y, tf.int32)
        y = tf.cast(y, tf.float32)
        y = sign * (2 ** (y - N))
        input_shape = input_layer.get_shape().as_list()
        new_layer = tf.cond(self.__is_training,
                            lambda:input_layer,
                            lambda:tf.reshape(y, [-1] + input_shape[1:]))
        return new_layer

    def add_quantize(self, N, input_name = None):
        input_layer = self.get_input(input_name)
        new_layer = self.make_quantize(N, input_layer)
        self.add_layer(new_layer)
    
    def add_switch(self, true_layer, false_layer):
        new_layer = tf.cond(self.__is_training,
                            lambda:true_layer,
                            lambda:false_layer)
        self.add_layer(new_layer)

    def add_rect_loss(self, name, gamma, alpha,
                      offset_y_list, offset_x_list, size_list, asp_list,
                      cls_layer_name, reg_layer_name,
                      cls_label_name, reg_label_name):
        pred_cls = self.get_input(cls_layer_name)
        pred_reg = self.get_input(reg_layer_name)
        pred_shape = pred_cls.get_shape().as_list()
        div_y = pred_shape[1]
        div_x = pred_shape[2]
        rect_ch = pred_shape[3]
        cls_num = pred_shape[4]
        
        label_cls = tf.placeholder(dtype = tf.int32,
                                   shape = [None, div_y, div_x, rect_ch]) # NOT one-hot
        assert(not cls_label_name in self.__label_dict.keys())
        self.__label_dict[cls_label_name] = label_cls
        anchored_label_reg = tf.placeholder(dtype = tf.float32,
                                            shape = [None, div_y, div_x, rect_ch, 4])
        assert(not reg_label_name in self.__label_dict.keys())
        self.__label_dict[reg_label_name] = anchored_label_reg
        anchor_ph = tf.placeholder(dtype = tf.float32,
                                   shape = [None, div_y, div_x, rect_ch, 4])
        self.__anchor_ph[reg_label_name] = anchor_ph
        
        #cls_valid = (label_cls >= 0)
        reg_valid = (label_cls > 0)
        reg_valid_num = tf.reduce_sum(tf.cast(reg_valid, tf.float32))
        anchored_label_reg = tf.boolean_mask(anchored_label_reg, reg_valid)
        anchor_ph = tf.boolean_mask(anchor_ph, reg_valid)
        pred_reg = tf.reshape(pred_reg, [-1, div_y, div_x, rect_ch, 4])
        pred_reg  = tf.boolean_mask(pred_reg, reg_valid)

        # loss
        loss_ty = tf.reduce_sum((tf.abs(pred_reg[:,0] - anchored_label_reg[:,0])))
        loss_tx = tf.reduce_sum((tf.abs(pred_reg[:,1] - anchored_label_reg[:,1])))
        loss_h  = tf.reduce_sum((tf.abs(pred_reg[:,2] - anchored_label_reg[:,2] )))
        loss_w  = tf.reduce_sum((tf.abs(pred_reg[:,3] - anchored_label_reg[:,3] )))
        zero = tf.constant(0.0)
        reg_loss = loss_ty + loss_tx + loss_h + loss_w
        reg_loss = tf.cond(reg_valid_num > 0.0, lambda: reg_loss / reg_valid_num, lambda: zero)
        assert(not reg_label_name in self.__loss_dict.keys())
        self.__loss_dict[reg_label_name]  = reg_loss
        
        
        label_cls_onehot = tf.one_hot(label_cls, depth = cls_num)
        pred_cls = tf.reshape(pred_cls, [-1, div_y, div_x, rect_ch, cls_num])
        p_t = (1.0 - pred_cls) + (2.0 * pred_cls - 1.0) * label_cls_onehot
        alpha_mat = (1.0 - alpha) / (cls_num - 1) + (alpha * cls_num - 1.0) / (cls_num - 1) * label_cls_onehot
        cls_loss_onehot = - alpha_mat * ((1.0 - p_t) ** gamma) * tf.log(p_t + 1e-5)
        cls_loss_vec = tf.reduce_sum(cls_loss_onehot, axis = [0,1,2,3])
        cls_loss = tf.reduce_sum(cls_loss_vec)
        '''
        focal loss論文に準拠するならば、easy-negativeは実質的にロスがないためポジティブなanchorの数で正規化すべき。
        ただ実際には画像内にポジティブ画像が全く無い場合も学習の対象としたいので、その場合は
        ポジ／ネガすべてのアンカー数で正規化する。その場合、ゼロ割はほとんど無いはずだが一応対策しておく。
        '''
        norm = tf.reduce_sum(tf.cast(label_cls > 0, tf.float32))
        neg_norm = tf.reduce_sum(tf.cast(label_cls >= 0, tf.float32))
        cls_loss = tf.cond(norm > 0.0, lambda: cls_loss / norm, lambda: tf.cond(neg_norm > 0.0, lambda: cls_loss / neg_norm, lambda: zero))
        assert(not label_cls_onehot in self.__loss_dict.keys())
        self.__loss_dict[cls_label_name]  = cls_loss
        base_anchor = make_anchor([div_y, div_x],
                                  offset_y_list = offset_y_list,
                                  offset_x_list = offset_x_list,
                                  size_list = size_list,
                                  asp_list = asp_list)
        self.__anchor[reg_label_name] = base_anchor.reshape(div_y,
                                                            div_x,
                                                            offset_y_list.size * offset_x_list.size * size_list.size * asp_list.size,
                                                            4)
        
    
    def get_anchor(self, name):
        return self.__anchor_ph[name], self.__anchor[name]
    
    '''        
    def __get_optimizer(self, loss, optimizer_type, learning_rate):
        if optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else:
            assert(0)
        return optimizer
    '''
    def create_feed_dict(self, input_image, is_training, label_dict = None):
        
        feed_dict = {}
        # input image
        feed_dict[self.__layer_list[0]] = input_image
        
        if label_dict is not None:
            for name, label in label_dict.items():
                feed_dict[self.__label_dict[name]] = label
        
        # for dropout
        if is_training is True:
            for dropout_idx in range(len(self.__dropout_ph_list)):
                feed_dict[self.__dropout_ph_list[dropout_idx]] = self.__dropout_val_list[dropout_idx]
        else:
            for dropout_idx in range(len(self.__dropout_ph_list)):
                feed_dict[self.__dropout_ph_list[dropout_idx]] = 1.0
        
        feed_dict[self.__is_training] = is_training
        return feed_dict
    
    def save_model(self, sess, saver, dst_dir_path, epoch = None, b = None, log_interval_sec = 1):#30 * 60):
        exec_save = False
        if self.__saved_time == None:
            # 初めて
            exec_save = True
        else:
            # ２回め以降だが、十分時間経過した
            if time.time() - self.__saved_time >= log_interval_sec:
                exec_save = True
        
        if exec_save:
            self.__saved_time = time.time()
            dst_name = "model"
            if epoch != None:
                dst_name = dst_name + "epoch{0:04d}".format(epoch)                
            if b != None:
                dst_name = dst_name + "_batch{}".format(b)
            dst_model_dir = os.path.join(dst_dir_path, dst_name)
            if not os.path.exists(dst_model_dir):
                os.makedirs(dst_model_dir)
            dst_model_path = os.path.join(dst_model_dir, "model.ckpt")
            saver.save(sess, dst_model_path)

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

