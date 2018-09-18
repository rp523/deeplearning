#coding:utf-8
import sys, os, shutil
import numpy as np
import json
from PIL import Image, ImageDraw

import storage
from common import fileio
from dataset_base import label_dict

class CityScapes:
    def __init__(self):
        s = storage.Storage()
        self.__data_path = s.dataset_path("cityscapes")
        self.__coarse_json = {}
    
    def get_coarse_label(self, label_type, index = None):
        tgt_dir = os.path.join(self.__data_path, "gtCoarse", "gtCoarse")
        assert(label_type == "train" or \
               label_type == "train_extra" or\
               label_type == "val")
        if not label_type in self.__coarse_json.keys():
            tgt_dir = os.path.join(tgt_dir, label_type)
            assert(os.path.exists(tgt_dir))
            json_path_list = fileio.get_file_list(tgt_dir = tgt_dir,
                                                  tgt_ext = ".json")
            self.__coarse_json[type] = json_path_list
        if index is None:
            index = np.random.randint(len(self.__coarse_json[type]))
        json_path = self.__coarse_json[type][index]
        info = json.load(open(json_path))
        h = info["imgHeight"]
        w = info["imgWidth"]
        obj_list = info["objects"]
        obj_num = len(obj_list)
        
        labels = np.empty(0).astype(np.int)
        rects = np.empty((0, 4)).astype(np.float)
        pixel_img = Image.fromarray(np.zeros((h, w)).astype(np.uint8))
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
        #pixel_img.resize((w//4,h//4)).show();exit()
        pixel_map = np.asarray(pixel_img)
        return labels, rects, pixel_map
        
if __name__ == "__main__":
    c = CityScapes()
    print(c.get_coarse_label("val"))
    