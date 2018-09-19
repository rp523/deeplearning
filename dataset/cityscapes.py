#coding:utf-8
import sys, os, shutil
import numpy as np
import json
from PIL import Image, ImageDraw
import re

import storage
from common import fileio
from dataset_base import label_dict

class CityScapes:
    def __init__(self):
        s = storage.Storage()
        data_path = s.dataset_path("cityscapes")
        self.__img_path_list, self.__json_path_list = self.__update_list(data_path)
        
    def __update_list(self, data_path):
        json_path_list = fileio.get_file_list(tgt_dir = data_path,
                                              tgt_ext = ".json")
        img_path_list_  = fileio.get_file_list(tgt_dir = data_path,
                                              tgt_ext = ".png",
                                              find_txt = "leftImg8bit.png")
        json_key_list = []
        for json_path in json_path_list:
            json_name = fileio.get_file_name(json_path)
            json_key_list.append(re.search("\D+\d+_\d+", json_name).group())
        img_key_list = []
        for img_path in img_path_list_:
            img_name = fileio.get_file_name(img_path)
            img_key_list.append(re.search("\D+\d+_\d+", img_name).group())
        img_path_list = []
        for json_key in json_key_list:
            idx = img_key_list.index(json_key)
            new_path = img_path_list_[idx]
            img_path_list.append(new_path)

        return img_path_list, json_path_list
    
    def get_one_data(self, index = None):
        if index is None:
            index = np.random.randint(len(self.__json_path_list))
        
        img_path = self.__img_path_list[index]
        print(img_path)
        rgb_img = Image.open(img_path)
        rgb_map = np.asarray(rgb_img)
        
        json_path = self.__json_path_list[index]
        info = json.load(open(json_path))
        h = info["imgHeight"]
        w = info["imgWidth"]
        obj_list = info["objects"]
        obj_num = len(obj_list)
        
        labels = np.empty(0).astype(np.int)
        rects = np.empty((0, 4)).astype(np.float)
        pixel_img = Image.fromarray(np.zeros((h, w)).astype(np.uint8))
        pixel_label_map = np.asarray(pixel_img)
        draw = ImageDraw.Draw(pixel_img)
        
        for obj in obj_list:
            label_name   = obj["label"]
            if not (label_name in label_dict.keys()):
                print(label_name)
            assert(label_name in label_dict.keys())
            label_value = label_dict[label_name]
            if label_value != 0:
                labels = np.append(labels, label_value)
                polygon = np.array(obj["polygon"])
                fill_val = label_value
                draw.polygon(xy = polygon.flatten().tolist(),
                             fill = fill_val,
                             outline = fill_val)
                x = polygon[:,0]
                y = polygon[:,1]
                x0 = np.min(x)
                y0 = np.min(y)
                x1 = np.max(x)
                y1 = np.max(y)
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, w - 1)
                y1 = min(y1, h - 1)
                rects = np.append(rects, np.array([x0 / w, y0 / h, x1 / w, y1 / h]).reshape(1, 4), axis = 0)
        if 1: #debug view
            
            draw = ImageDraw.Draw(pixel_img)
            dw, dh = pixel_img.size
            for i in range(rects.shape[0]):
                rect = rects[i]
                x0 = int(rect[0] * dw)
                y0 = int(rect[1] * dh)
                x1 = int(rect[2] * dw)
                y1 = int(rect[3] * dh)
                draw.rectangle([x0, y0, x1, y1], outline = 255)
                for k, v in label_dict.items():
                    if v == labels[i]:
                        draw.text([max(0, min(x0, dw - 1)),
                                   max(0, min(y0, dw - 1))],
                                  k,
                                  fill = 255)
                        break
            pixel_img.resize((pixel_img.size[0]//2,pixel_img.size[1]//2)).show()
            rgb_img.resize((rgb_img.size[0]//2,rgb_img.size[1]//2)).show()
            exit()

        return rgb_map, labels, rects, pixel_label_map
        
if __name__ == "__main__":
    c = CityScapes()
    print(c.get_one_data())
    