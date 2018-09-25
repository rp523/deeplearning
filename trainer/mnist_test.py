#coding: utf-8
import os, sys
sys.path.append("../")
from dataset.mnist import MNIST
from model.network import ImageNetwork

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
    network.add_softmax()
    network.show()