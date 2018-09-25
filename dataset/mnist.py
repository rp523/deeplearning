#coding:utf-8
import sys, os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

sys.path.append("../")
import storage
from common import fileio

class MNIST:
    
    def __init__(self, one_hot = True):
        s = storage.Storage()
        data_path = s.dataset_path("mnist")
        assert(os.path.exists(data_path))

        with open(os.path.join(data_path, "train-images.idx3-ubyte"), "rb") as f:
            self.__train_images = np.array(list(f.read())[16:]).astype(np.uint8).reshape(-1, 28, 28, 1)
        with open(os.path.join(data_path, "t10k-images.idx3-ubyte"), "rb") as f:
            self.__valid_images = np.array(list(f.read())[16:]).astype(np.uint8).reshape(-1, 28, 28, 1)
        with open(os.path.join(data_path, "train-labels.idx1-ubyte"), "rb") as f:
            self.__train_labels = np.array(list(f.read())[8:]).astype(np.uint8).flatten()
        with open(os.path.join(data_path, "t10k-labels.idx1-ubyte"), "rb") as f:
            self.__valid_labels = np.array(list(f.read())[8:]).astype(np.uint8).flatten()
        
        if one_hot is True:
            train_labels = np.zeros((self.__train_labels.size, 10)).astype(np.uint8)
            valid_labels = np.zeros((self.__valid_labels.size, 10)).astype(np.uint8)
            train_labels[self.__train_labels] = 1
            valid_labels[self.__valid_labels] = 1
            self.__train_labels = self.__train_labels
            self.__valid_labels = self.__valid_labels
        
        assert(self.__train_images.shape[0] == self.__train_labels.shape[0])
        assert(self.__valid_images.shape[0] == self.__valid_labels.shape[0])
    
    def get_data_num(self, data_type):
        if   data_type == "train":
            return self.__train_labels.size
        elif data_type == "val":
            return self.__valid_labels.size
        else:
            assert(0)
            
    def get_valid_num(self):
        return self.__valid_labels.size
    
    def get_data(self, data_type, index):
        if   data_type == "train":
            images = self.__train_images
            labels = self.__train_labels
        elif data_type == "val":
            images = self.__valid_images
            labels = self.__valid_labels
        else:
            assert(0)
        
        ret_images = images[index]
        ret_labels = labels[index]
        
        return ret_images, ret_labels
            
if __name__ == "__main__":
    m = MNIST()
    num = m.get_data_num("val")
    img, label = m.get_data("val", np.random.randint(num))
    from PIL import Image
    print(label)
    Image.fromarray(img.astype(np.uint8).reshape(28, 28)).show()
    print("Done.")
