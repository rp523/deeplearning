#coding: utf-8
import os, sys, subprocess
import numpy as np
if __name__ == "__main__":
    sys.path.append(subprocess.getoutput("pwd"))
from model.network import ImageNetwork



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
                network.add_softmax()
        else:
            break

    network.add_loss(loss_type = "cross_entropy", name = "edge")
    return network

def main():
    network = dedge_net(img_h = 256,
                        img_w = 512,
                        img_ch = 3,
                        out_h = 32)
    print(network.show_all())

if __name__ == "__main__":
    main()
    print("Done.")