#coding: utf-8
import os, sys, subprocess
import numpy as np
from tqdm import tqdm
import tensorflow as tf
if __name__ == "__main__":
    sys.path.append(subprocess.getoutput("pwd"))
from model.network import ImageNetwork
from dataset.bdd100k import BDD100k
from datetime import datetime

def dedge_net(img_h,
              img_w,
              img_ch,
              out_h):
    
    network = ImageNetwork(img_h, img_w, img_ch, random_seed = None)
    while True:
        last_shape = network.get_input(None).get_shape().as_list()
        
        if last_shape[2] > 1:
            if last_shape[2] > 2:
                if last_shape[1] > out_h:
                    stride_y = 2
                else:
                    stride_y = 1
                network.add_conv_batchnorm_act(ImageNetwork.FilterParam(3, 3, stride_y, 2, True), 32, "relu")
            else:
                network.add_conv(ImageNetwork.FilterParam(3, 3, stride_y, 2, True), 3)
                network.add_batchnorm()
                network.add_softmax(name = "output")
        else:
            break

    return network

def dedge_main():
    h_pix = 128
    w_pix = 256
    epoch_num = 100
    batch_size = 4
    lr = 1E-2
    network = dedge_net(img_h = h_pix,
                        img_w = w_pix,
                        img_ch = 3,
                        out_h = h_pix)
    network.add_loss(loss_type = "cross_entropy", input_name = "output", name = "edge", masking = True)
    loss = network.get_loss_dict()["edge"]
    opt  = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
    bdd = BDD100k()
    train_type = "debug"
    val_type = "debug"
    log_interval_sec = 30 * 60
    restore_path = None#"/home/isgsktyktt/workspace/deeplearning/result_20190212_074345/learned_model/modelepoch0003_batch16243"
    result_dir = "result_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(result_dir, "learned_model")
    with tf.Session() as sess:

        # restore
        saver = tf.train.Saver()
        if restore_path:
            ckpt = tf.train.get_checkpoint_state(restore_path)
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        
        # only evaluation
        if 0:
            for i in tqdm(range(bdd.get_sample_num(val_type))):
                rgb_arr, drivable_edge = bdd.get_drivable_edge_data(data_type = val_type, index = i, h_pix = h_pix, w_pix = w_pix)
                feed_dict = network.create_feed_dict(input_image = np.reshape(rgb_arr, [-1] + list(rgb_arr.shape)),
                                                    is_training = False)
                output = sess.run(network.get_layer("output"), feed_dict = feed_dict)
                output = output.reshape(output.shape[1], 3)
                output = np.delete(output, 1, axis = 1)
                dst_dir_path = os.path.join(result_dir, "val")
                if not os.path.exists(dst_dir_path):
                    os.makedirs(dst_dir_path)
                bdd.summary_drivable_edge_data(rgb_arr, output).save(os.path.join(dst_dir_path, val_type + "_{}.png".format(i)))

        for epoch in range(epoch_num):
            for i in tqdm(range(bdd.get_sample_num(train_type) // batch_size)):
                # make batched-dataset
                img_batch = np.zeros((batch_size, h_pix, w_pix, 3)).astype(np.float)
                lbl_batch = np.zeros((batch_size, h_pix,     1, 3)).astype(np.float)
                val_batch = np.zeros((batch_size, h_pix,     1, 3)).astype(np.bool)
                for b in range(batch_size):
                    rgb_arr, drivable_edge = bdd.get_drivable_edge_data(data_type = train_type, h_pix = h_pix, w_pix = w_pix)
                    img_batch[b] = rgb_arr

                    valid_row = (drivable_edge[:,0] > 0.0)
                    for ch in range(3):
                        val_batch[b, :, 0, ch] = valid_row

                    drivable_edge[drivable_edge < 0] = 0.5  # 計算後にマスクかけるので、変な値は入れちゃダメ
                    lbl_batch[b, :, 0, 0] = drivable_edge[:,0]
                    lbl_batch[b, :, 0, 2] = drivable_edge[:,1]
                    lbl_batch[b, :, 0, 1] = 1.0 - (lbl_batch[b, :, 0, 0] + lbl_batch[b, :, 0, 2])

                feed_dict = network.create_feed_dict(input_image = img_batch,
                                                     is_training = True,
                                                     label_dict = {"edge" : lbl_batch,
                                                                   "edge_mask": val_batch})
                assert((lbl_batch[val_batch] >= 0.0).all())
                print(sess.run([opt, loss], feed_dict = feed_dict))
                network.save_model(sess, saver, model_path, epoch, i, log_interval_sec)
        network.save_model(sess, saver, model_path)
    from PIL import Image
    Image.fromarray(rgb_arr.astype(np.uint8)).show()

if __name__ == "__main__":
    dedge_main()
    print("Done.")