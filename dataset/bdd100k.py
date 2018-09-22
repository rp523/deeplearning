#coding:utf-8
import sys, os, shutil
import numpy as np
import json
from PIL import Image, ImageDraw
import re

sys.path.append("../")
import storage
from common import fileio
from dataset_base import label_dict

class BDD100k:
    def __init__(self):
        s = storage.Storage()
        self.__data_path = s.dataset_path("bdd100k")
        self.__rgb_path_dict = {}
        self.__json_path_dict = {}
        self.__rgb_h, self.__rgb_w = 720, 1280

    def __update_list(self, data_type):
        assert((data_type == "train") or \
               (data_type == "val"  ) or \
               (data_type == "test" ))
        if not data_type in self.__json_path_dict.keys():
            json_dir_path = os.path.join(self.__data_path, "labels", data_type)
            if os.path.exists(json_dir_path):
                json_path_list_ = fileio.get_file_list(tgt_dir = json_dir_path,
                                                       tgt_ext = ".json")
            rgb_dir_path = os.path.join(self.__data_path, "images", data_type)
            if os.path.exists(rgb_dir_path):
                rgb_path_list_ = fileio.get_file_list(tgt_dir = rgb_dir_path,
                                                     tgt_ext = ".jpg")
            
            json_key_list = []
            rgb_key_list = []
            for json_path in json_path_list_:
                key = fileio.get_file_name(json_path)[:-len(".json")]
                json_key_list.append(key)
            for rgb_path in rgb_path_list_:
                key = fileio.get_file_name(rgb_path)[:-len(".jpg")]
                rgb_key_list.append(key)
            
            json_path_list = []
            rgb_path_list = []
            for j in range(len(json_key_list)):
                json_key = json_key_list[j]
                if json_key in rgb_key_list:
                    i = rgb_key_list.index(json_key)
                    json_path_list.append(json_path_list_[j])
                    rgb_path_list.append(rgb_path_list_[i])
            assert(len(json_path_list) == len(rgb_path_list))
            self.__rgb_path_dict[data_type] = rgb_path_list
            self.__json_path_dict[data_type] = json_path_list
    
    def get_sample_num(self, data_type):
        self.__update_list(data_type)
        return len(self.__rgb_path_dict[data_type])
    
    def get_one_data(self, data_type, index = None):
        self.__update_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__json_path_dict[data_type]))
        rgb_path = self.__rgb_path_dict[data_type][index]
        rgb_map = np.asarray(Image.open(rgb_path))
        
        json_path = self.__json_path_dict[data_type][index]

        info = json.load(open(json_path))
        obj_list = info["labels"]
        obj_num = len(obj_list)
        
        rect_labels = np.empty(0).astype(np.int)
        rects = np.empty((0, 4)).astype(np.float)
        poly_labels = np.empty(0).astype(np.int)
        polygons = []
        
        for obj in obj_list:
            label_name   = obj["category"]
            if not (label_name in label_dict.keys()):
                print(label_name)
            assert(label_name in label_dict.keys())
            label_value = label_dict[label_name]
            if label_value != 0:
                if "box2d" in obj.keys():
                    box_dict = obj["box2d"]
                    x0 = box_dict["x1"] / self.__rgb_w
                    x1 = box_dict["x2"] / self.__rgb_w
                    y0 = box_dict["y1"] / self.__rgb_h
                    y1 = box_dict["y2"] / self.__rgb_h
                    rects = np.append(rects, np.array([x0, y0, x1, y1]).reshape(1, 4), axis = 0)
                    rect_labels = np.append(rect_labels, label_value)
                elif "poly2d" in obj.keys():
                    polygon = np.array(obj["poly2d"][0]["vertices"]).reshape(-1, 2)
                    if polygon.shape[0] >= 3:
                        polygon = polygon / np.array([self.__rgb_w, self.__rgb_h])
                        polygon = polygon.flatten()
                        polygons.append(polygon)
                    poly_labels = np.append(poly_labels, label_value)
    
        if 0: #debug view
            self.show_frame_data(rgb_map, rect_labels, rects, poly_labels, polygons).show()
            exit()

        return rgb_map, rect_labels, rects, poly_labels, polygons
    
    def show_frame_data(self, rgb_map, rect_labels, rects, poly_labels, polygons):
        rgb_img = Image.fromarray(rgb_map)
        draw = ImageDraw.Draw(rgb_img, "RGBA")
        dw, dh = rgb_img.size
        red   = (255,   0,   0, 128)
        blue  = (  0,   0, 255, 64)
        for i in range(rects.shape[0]):
            rect = rects[i]
            x0 = (rect[0] * dw)
            y0 = (rect[1] * dh)
            x1 = (rect[2] * dw)
            y1 = (rect[3] * dh)
            draw.rectangle([x0, y0, x1, y1], outline = red)
            for k, v in label_dict.items():
                if v == rect_labels[i]:
                    draw.text([max(0, min(x0, dw - 1)),
                               max(0, min(y0, dh - 1))],
                               k,
                               fill = red)
                    break
        for i in range(len(polygons)):
            polygon = (polygons[i].reshape(-1, 2) * [dw , dh])
            draw.polygon(polygon.flatten().tolist(), fill = blue)
            for k, v in label_dict.items():
                if v == poly_labels[i]:
                    draw.text([max(0, min(dw - 1, np.average(polygon[:, 0]))),
                               max(0, min(dh - 1, np.average(polygon[:, 1])))],
                               k,
                               fill = blue)
                    break
        #rgb_img.show()
        return rgb_img
        
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
                            with open(dst_path, 'w') as fout:
                                json.dump(cont, fout, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
                            read_flg = False
                        else:
                            if read_flg:
                                out = out + line
                        line = fin.readline()
if __name__ == "__main__":
    b = BDD100k()
    from tqdm import tqdm
    data_type = "val"
    dst_dir_path = r"G:\dataset\bdd100k\tmp"

    dst_dir = os.path.join(dst_dir_path, data_type)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i in tqdm(range(b.get_sample_num(data_type))):
        rgb_map, rect_labels, rects, poly_labels, polygons = b.get_one_data(data_type, i)
        b.show_frame_data(rgb_map, rect_labels, rects, poly_labels, polygons).save( \
            os.path.join(dst_dir, "{0:06d}.png".format(i)))
    print("Done.")

    