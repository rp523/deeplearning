#coding:utf-8
import sys, os, shutil
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import re

import storage
from common import fileio
from dataset_base import label_dict

class CityScapes:
    def __init__(self):
        s = storage.Storage()
        data_path = s.dataset_path("cityscapes")
        self.__img_path_list, self.__json_path_list = self.__update_list(data_path)
        self.__debug = True
        
    def __update_list(self, data_path):
        json_path_list_ = fileio.get_file_list(tgt_dir = os.path.join(data_path, "gtFine_trainvaltest"), # Fineだけ,
                                               exclude_path_txt = os.sep + "test" + os.sep,
                                               tgt_ext = ".json")
        img_path_list_  = fileio.get_file_list(tgt_dir = os.path.join(data_path, "leftImg8bit_trainvaltest"),
                                               tgt_ext = ".png",
                                               name_txt = "leftImg8bit.png")
        json_key_list = []
        for json_path in json_path_list_:
            name = fileio.get_file_name(json_path)
            idx = name.find("_")
            idx = idx + 1 + name[idx + 1:].find("_")
            idx = idx + 1 + name[idx + 1:].find("_")
            key = name[:idx]
            json_key_list.append(key)
        img_key_list = []
        for img_path in img_path_list_:
            name = fileio.get_file_name(img_path)
            idx = name.find("_")
            idx = idx + 1 + name[idx + 1:].find("_")
            idx = idx + 1 + name[idx + 1:].find("_")
            key = name[:idx]
            img_key_list.append(key)

        img_path_list = []
        json_path_list = []
        for j in range(len(json_key_list)):
            json_key = json_key_list[j]
            if json_key in img_key_list:
                idx = img_key_list.index(json_key)
                img_path_list.append(img_path_list_[idx])
                json_path_list.append(json_path_list_[j])

        return img_path_list, json_path_list
    
    def get_one_data(self, tgt_label_name_list = None, index = None):
        if index is None:
            rand_max = min(len(self.__json_path_list), len(self.__img_path_list))
            index = np.random.randint(rand_max)
        
        img_path = self.__img_path_list[index]
        rgb_img = Image.open(img_path)
        rgb_map = np.asarray(rgb_img)
        
        json_path = self.__json_path_list[index]
        #print(img_path)
        #rint(json_path)
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
        
        for o in range(len(obj_list)):
            obj = obj_list[o]
            label_name   = obj["label"]
            if not (label_name in label_dict.keys()):
                print(label_name)
            assert(label_name in label_dict.keys())
            if tgt_label_name_list is not None:
                if not label_name in tgt_label_name_list:
                    continue
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
                rect = np.array([x0 / w, y0 / h, x1 / w, y1 / h]).reshape(1, 4)
                rects = np.append(rects, rect, axis = 0)
            
            if self.__debug: #debug view
                rgb_draw = ImageDraw.Draw(rgb_img)
                dw, dh = rgb_img.size
                if label_value != 0:
                    rect = rects[-1]
                    x0 = int(rect[0] * dw)
                    y0 = int(rect[1] * dh)
                    x1 = int(rect[2] * dw)
                    y1 = int(rect[3] * dh)
                    rgb_draw.rectangle([x0, y0, x1, y1], outline = (255, 255, 255))
                rgb_draw.polygon(np.array(obj["polygon"]).flatten().tolist(), outline = (255,0,0))
                rgb_draw.text([max(0, min(x0, dw - 1)),
                               max(0, min(y0, dw - 1))],
                               label_name,
                               fill = (255, 255, 255),
                               font = ImageFont.truetype("arial.ttf", size = 16))
                #print(label_name)
        if self.__debug:
            rgb_img.show()
            exit()

        return rgb_map, labels, rects, pixel_label_map
        
if __name__ == "__main__":
    c = CityScapes()
    (c.get_one_data(["car", "truck", "bus", "trailer", "bicycle", "person", "rider"]))
    