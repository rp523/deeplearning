#coding:utf-8
import sys, os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

sys.path.append("../")
import storage
from common import fileio
from dataset_base import *

bdd_seg_dict = {}
bdd_seg_dict["sidewalk"] = 1
bdd_seg_dict["building"] = 2
bdd_seg_dict["wall"] = 3
bdd_seg_dict["fence"] = 4
bdd_seg_dict["pole"] = 5
bdd_seg_dict["traffic light"] = 6
bdd_seg_dict["billboard"] = 7
bdd_seg_dict["vegetation"] = 8
bdd_seg_dict["ground"] = 9
bdd_seg_dict["sky"] = 10
bdd_seg_dict["person"] = 11
bdd_seg_dict["rider"] = 12
bdd_seg_dict["car"] = 13
bdd_seg_dict["truck"] = 14
bdd_seg_dict["bus"] = 15
bdd_seg_dict["train"] = 16
bdd_seg_dict["motorcycle"] = 17
bdd_seg_dict["bicycle"] = 18
bdd_seg_dict["out of eval"] = 255

class BDD100k:
    
    def __init__(self):
        s = storage.Storage()
        self.__data_path = s.dataset_path("bdd100k")
        self.__rgb_path_dict = {}
        self.__json_path_dict = {}
        self.__area_path_dict = {}
        self.__seg_path_dict = {}
        self.__segimg_path_dict = {}
        self.__rgb_h, self.__rgb_w = 720, 1280

    def __update_seg_list(self, data_type):
        assert((data_type == "train") or \
               (data_type == "val"  ) or \
               (data_type == "test" ))
        if not data_type in self.__seg_path_dict.keys():
            seg_dir_path = os.path.join(self.__data_path, "seg", "labels", data_type)
            if os.path.exists(seg_dir_path):
                seg_path_list_ = fileio.get_file_list(tgt_dir = seg_dir_path,
                                                     tgt_ext = ".png")
            segimg_dir_path = os.path.join(self.__data_path, "seg", "images", data_type)
            if os.path.exists(segimg_dir_path):
                segimg_path_list_ = fileio.get_file_list(tgt_dir = segimg_dir_path,
                                                        tgt_ext = ".jpg")
            
            seg_key_list = []
            segimg_key_list = []
            for seg_path in seg_path_list_:
                name = fileio.get_file_name(seg_path)
                key  = name[:name.find("_")]
                seg_key_list.append(key)
            for img_path in segimg_path_list_:
                name = fileio.get_file_name(img_path)
                key  = name[:name.rfind(".")]
                segimg_key_list.append(key)
            
            seg_path_list = []
            segimg_path_list = []
            for s in range(len(seg_key_list)):
                seg_key = seg_key_list[s]
                if seg_key in segimg_key_list:
                    seg_path_list.append(seg_path_list_[s])
                    segimg_path_list.append(segimg_path_list_[segimg_key_list.index(seg_key)])
            assert(len(seg_path_list) > 0)
            assert(len(seg_path_list) == len(segimg_path_list))

            self.__seg_path_dict[data_type]    = seg_path_list
            self.__segimg_path_dict[data_type] = segimg_path_list
                
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
            area_dir_path = os.path.join(self.__data_path, "drivable_maps", data_type)
            if os.path.exists(area_dir_path):
                area_path_list_ = fileio.get_file_list(tgt_dir = area_dir_path,
                                                       tgt_ext = ".png")
            assert(len(json_path_list_) > 0)
            assert(len(rgb_path_list_) > 0)
            assert(len(area_path_list_) > 0)

            json_key_list = []
            rgb_key_list  = []
            area_key_list = []
            for json_path in json_path_list_:
                key = fileio.get_file_name(json_path)[:-len(".json")]
                json_key_list.append(key)
            for rgb_path in rgb_path_list_:
                key = fileio.get_file_name(rgb_path)[:-len(".jpg")]
                rgb_key_list.append(key)
            for area_path in area_path_list_:
                key = fileio.get_file_name(area_path)[:-len("_drivable_id.png")]
                area_key_list.append(key)
            assert(len(json_key_list) > 0)
            assert(len(rgb_key_list) > 0)
            assert(len(area_key_list) > 0)
            
            json_path_list = []
            rgb_path_list  = []
            area_path_list = []
            for j in range(len(json_key_list)):
                json_key = json_key_list[j]
                if json_key in rgb_key_list:
                    if json_key in area_key_list:
                        json_path_list.append(json_path_list_[j])
                        i = rgb_key_list.index(json_key)
                        rgb_path_list.append(rgb_path_list_[i])
                        i = area_key_list.index(json_key)
                        area_path_list.append(area_path_list_[i])
            assert(len(json_path_list) == len(rgb_path_list))
            assert(len(json_path_list) == len(area_path_list))
            assert(len(json_path_list) > 0)
            assert(len(rgb_path_list) > 0)
            assert(len(area_path_list) > 0)
            self.__rgb_path_dict[data_type]  = rgb_path_list
            self.__json_path_dict[data_type] = json_path_list
            self.__area_path_dict[data_type] = area_path_list
    
    def get_sample_num(self, data_type):
        self.__update_list(data_type)
        return len(self.__rgb_path_dict[data_type])
    
    def get_seg_sample_num(self, data_type):
        self.__update_seg_list(data_type)
        return len(self.__seg_path_dict[data_type])
    
    def get_seg_data(self, data_type, index = None):
        self.__update_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__seg_path_dict[data_type]))
        seg_path = self.__seg_path_dict[data_type][index]
        seg_arr = np.asarray(Image.open(seg_path))

        rgb_path = self.__segimg_path_dict[data_type][index]
        rgb_arr = np.asarray(Image.open(rgb_path))
        return rgb_arr, seg_arr
    
    def summary_seg_data(self, rgb_arr, seg_arr):
        amp = 4
        font_path = "arial.ttf"
        font_size = 24
        font = ImageFont.truetype(font_path, font_size)
        
        col = [0,0,255]
        sum_arr = rgb_arr.copy().astype(np.int)
        for v in [16]:
            fill_arr = rgb_arr.copy().astype(np.int)
            fill_arr[seg_arr == v] = col
            sum_arr = ((amp - 1) * sum_arr + fill_arr) // amp
        
        sum_arr = sum_arr.astype(np.uint8)
        sum_img = Image.fromarray(sum_arr)
        draw = ImageDraw.Draw(sum_img, "RGB")
        cnt = 0
        for k, v in bdd_seg_dict.items():
            draw.text([0, cnt],
                       k,
                       fill = tuple(col),
                       font = font)
            cnt += font_size
        return sum_img

    def get_area_data(self, data_type, index = None):
        self.__update_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__area_path_dict[data_type]))
        rgb_path = self.__rgb_path_dict[data_type][index]
        rgb_arr = np.asarray(Image.open(rgb_path))

        area_path = self.__area_path_dict[data_type][index]
        area_arr = np.asarray(Image.open(area_path))
        assert(len(area_dict.values()) >= np.max(area_arr))
        return rgb_arr, area_arr
    
    def summary_area_data(self, rgb_arr, area_arr):
        amp = 4
        font_path = r"C:\Windows\Fonts\Myrica.TTC"
        font_size = 24
        font = ImageFont.truetype(font_path, font_size)

        sum_arr = rgb_arr.copy().astype(np.int)
        for k, v in area_dict.items():
            fill_img = rgb_arr.copy().astype(np.int)
            col = np.zeros(3).astype(np.int)
            col[v - 1] = 255
            fill_img[area_arr == v] = col
            sum_arr = ((amp - 1) * sum_arr + fill_img) // amp
        
        rise_x = np.empty((area_arr.shape[0], area_arr.shape[1])).astype(np.int)
        rise_y = np.empty((area_arr.shape[0], area_arr.shape[1])).astype(np.int)
        rise_x[:, :] = np.arange(area_arr.shape[1]).reshape(1, -1)
        rise_y[:, :] = np.arange(area_arr.shape[0]).reshape(-1, 1)
        
        sum_arr = sum_arr.astype(np.uint8)
        sum_img = Image.fromarray(sum_arr)
        draw = ImageDraw.Draw(sum_img, "RGB")
        for k, v in area_dict.items():
            fill_arr = np.zeros((area_arr.shape[0], area_arr.shape[1])).astype(np.float)
            fill_arr[area_arr == v] = 1
            col = np.zeros(3).astype(np.uint8)
            col[v - 1] = 255
            fill_x = rise_x * fill_arr
            text_x = np.average(fill_x[fill_x > 0])
            fill_y = rise_y * fill_arr
            text_y = np.average(fill_y[fill_y > 0])
            draw.text([text_x, text_y],
                       k,
                       fill = tuple(col.tolist()),
                       font = font)
        return sum_img

    def get_vertices_data(self, data_type, index = None):
        self.__update_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__json_path_dict[data_type]))
        rgb_path = self.__rgb_path_dict[data_type][index]
        rgb_arr = np.asarray(Image.open(rgb_path))
        
        json_path = self.__json_path_dict[data_type][index]

        info = json.load(open(json_path))
        obj_list = info["labels"]
        
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
            self.summary_vertices_data(rgb_arr, rect_labels, rects, poly_labels, polygons).show()
            exit()

        return rgb_arr, rect_labels, rects, poly_labels, polygons
    
    def summary_vertices_data(self, rgb_arr, rect_labels, rects, poly_labels, polygons):
        font_path = r"C:\Windows\Fonts\Myrica.TTC"
        font_size = 16
        font = ImageFont.truetype(font_path, font_size)
        red   = (255,   0,   0, 128)
        blue  = (  0,   0, 255,  64)
        white = (255, 255, 255, 255)

        rgb_img = Image.fromarray(rgb_arr)
        draw = ImageDraw.Draw(rgb_img, "RGBA")
        dw, dh = rgb_img.size
        for i in range(rects.shape[0]):
            rect = rects[i]
            x0 = (rect[0] * dw)
            y0 = (rect[1] * dh)
            x1 = (rect[2] * dw)
            y1 = (rect[3] * dh)
            draw.rectangle([x0, y0, x1, y1], outline = red)
            for k, v in label_dict.items():
                if v == rect_labels[i]:
                    draw.text([max(0, min(dw - 1, x0)),
                               max(0, min(dh - 1, y0 - font_size))],
                               k,
                               fill = red,
                               font = font)
                    break
        for i in range(len(polygons)):
            polygon = (polygons[i].reshape(-1, 2) * [dw , dh])
            draw.polygon(polygon.flatten().tolist(), fill = blue)
            for k, v in label_dict.items():
                if v == poly_labels[i]:
                    draw.text([max(0, min(dw - 1, np.average(polygon[:, 0]))),
                               max(0, min(dh - 1, np.average(polygon[:, 1])) - font_size)],
                               k,
                               fill = white,
                               font = font)
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
                
    def list_all_seg_val(self):
        seg_val = []
        for data_type in ["val", "train"]:
            for seg_path in fileio.get_file_list(tgt_dir = os.path.join(self.__data_path, "seg", "labels", data_type),
                                                 tgt_ext = ".png"):
                img = Image.open(seg_path)
                for new_val in np.unique(np.asarray(img)):
                    if not new_val in seg_val:
                        seg_val.append(new_val)
        for val in seg_val:
            print(val)
                
    def split_json(self):
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
                        
def make_seg_summary_img(data_type):
    b = BDD100k()
    from tqdm import tqdm
    dst_dir_path = os.path.join(storage.Storage().dataset_path("bdd100k"), "seg_test")

    dst_dir = os.path.join(dst_dir_path, data_type)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i in tqdm(range(b.get_seg_sample_num(data_type))):
        rgb_arr, seg_arr = b.get_seg_data(data_type, i)
        b.summary_seg_data(rgb_arr, seg_arr).save( \
            os.path.join(dst_dir, "{0:06d}.png".format(i)))

def make_area_summary_img(data_type):
    b = BDD100k()
    from tqdm import tqdm
    dst_dir_path = os.path.join(storage.Storage().dataset_path("bdd100k"), "area_test")

    dst_dir = os.path.join(dst_dir_path, data_type)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i in tqdm(range(b.get_sample_num(data_type))):
        rgb_arr, area_arr = b.get_area_data(data_type, i)
        b.summary_area_data(rgb_arr, area_arr).save( \
            os.path.join(dst_dir, "{0:06d}.png".format(i)))

def make_vertices_summary_img(data_type):
    b = BDD100k()
    from tqdm import tqdm
    dst_dir_path = os.path.join(storage.Storage().dataset_path("bdd100k"), "vertices_test")

    dst_dir = os.path.join(dst_dir_path, data_type)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i in tqdm(range(b.get_sample_num(data_type))):
        rgb_arr, rect_labels, rects, poly_labels, polygons = b.get_vertices_data(data_type, i)
        b.summary_vertices_data(rgb_arr, rect_labels, rects, poly_labels, polygons).save( \
            os.path.join(dst_dir, "{0:06d}.png".format(i)))
if __name__ == "__main__":
    make_seg_summary_img("val")
    print("Done.")
