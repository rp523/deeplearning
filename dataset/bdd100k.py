#coding:utf-8
import sys, os, shutil
import numpy as np
import json
from PIL import Image, ImageDraw
import re

import storage
from common import fileio
from dataset_base import label_dict

class BDD100k:
    def __init__(self):
        s = storage.Storage()
        self.__data_path = s.dataset_path("bdd100k")
        self.__img_path_dict = {}
        self.__json_path_dict = {}
        self.__img_h, self.__img_w = 720, 1280
        self.__debug = False

    def __update_list(self, data_type):
        assert((data_type == "train") or \
               (data_type == "val"  ) or \
               (data_type == "test" ))
        if not data_type in self.__json_path_dict.keys():
            json_dir_path = os.path.join(self.__data_path, "labels", data_type)
            if os.path.exists(json_dir_path):
                json_path_list_ = fileio.get_file_list(tgt_dir = json_dir_path,
                                                       tgt_ext = ".json")
            img_dir_path = os.path.join(self.__data_path, "images", data_type)
            if os.path.exists(img_dir_path):
                img_path_list_ = fileio.get_file_list(tgt_dir = img_dir_path,
                                                     tgt_ext = ".jpg")
            
            json_key_list = []
            img_key_list = []
            for json_path in json_path_list_:
                key = fileio.get_file_name(json_path)[:-len(".json")]
                json_key_list.append(key)
            for img_path in img_path_list_:
                key = fileio.get_file_name(img_path)[:-len(".jpg")]
                img_key_list.append(key)
            
            json_path_list = []
            img_path_list = []
            for j in range(len(json_key_list)):
                json_key = json_key_list[j]
                if json_key in img_key_list:
                    i = img_key_list.index(json_key)
                    json_path_list.append(json_path_list_[j])
                    img_path_list.append(img_path_list_[i])
            self.__img_path_dict[data_type] = img_path_list
            self.__json_path_dict[data_type] = json_path_list
    
    def get_one_data(self, data_type, index = None):
        self.__update_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__json_path_dict[data_type]))
        img_path = self.__img_path_dict[data_type][index]
        rgb_img = Image.open(img_path)
        rgb_map = np.asarray(rgb_img)
        
        json_path = self.__json_path_dict[data_type][index]

        info = json.load(open(json_path))
        obj_list = info["labels"]
        obj_num = len(obj_list)
        
        labels = np.empty(0).astype(np.int)
        rects = np.empty((0, 4)).astype(np.float)
        
        for obj in obj_list:
            label_name   = obj["category"]
            if not (label_name in label_dict.keys()):
                print(label_name)
            assert(label_name in label_dict.keys())
            label_value = label_dict[label_name]
            if label_value != 0:
                labels = np.append(labels, label_value)
                if "box2d" in obj.keys():
                    box_dict = obj["box2d"]
                    x0 = box_dict["x1"] / self.__img_w
                    x1 = box_dict["x2"] / self.__img_w
                    y0 = box_dict["y1"] / self.__img_h
                    y1 = box_dict["y2"] / self.__img_h
                    rects = np.append(rects, np.array([x0, y0, x1, y1]).reshape(1, 4), axis = 0)
        if self.__debug: #debug view
            rgb_img = rgb_img.resize((rgb_img.size[0]//2, rgb_img.size[1]//2))
            draw = ImageDraw.Draw(rgb_img)
            dw, dh = rgb_img.size
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
            rgb_img.show()

        return rgb_map, labels, rects
    
    def list_new_category(self):
        bdd_cat = []
        for json_path in fileio.get_file_list(tgt_dir = os.path.join(self.__data_path, "labels"),
                                              name_txt = "val",
                                              tgt_ext = ".json"):
            imgs = json.loads(open(json_path, "r").read())
            for img in imgs:
                for obj in img["labels"]:
                    if not obj["category"] in bdd_cat:
                        bdd_cat.append(obj["category"])
        for bdd in bdd_cat:
            if not bdd in label_dict.keys():
                print(bdd)
                
    def split_json(self):
        import re
        for data_type in ["val", "train"]:
            dst_dir = os.path.join(self.__data_path, "labels", data_type)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for json_path in fileio.get_file_list(tgt_dir = os.path.join(self.__data_path, "labels"),
                                                  name_txt = data_type,
                                                  tgt_ext = ".json"):
                with open(json_path, "r") as fin:
                    line = fin.readline()
                    read_flg = False
                    while line:
                        if line == "    {\n":
                            # start reading
                            read_flg = True
                            out = line
                        elif line.find("    }") == 0:
                            # end reading
                            out = out + line.replace(",", "")
                            cont = json.loads(out)
                            dst_path = os.path.join(dst_dir, cont["name"][:-len(".jpg")] + ".json")
                            print(dst_path)
                            with open(dst_path, 'w') as fout:
                                json.dump(cont, fout, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
                            read_flg = False
                        else:
                            if read_flg:
                                out = out + line
                        line = fin.readline()
if __name__ == "__main__":
    b = BDD100k()
    print(b.get_one_data("val"))
    print("Done.")

    