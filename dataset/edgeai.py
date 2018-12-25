#coding:utf-8
import sys, os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

sys.path.append("../")
import storage
from common import fileio
from dataset.__init__ import Dataset

class EdgeAI(Dataset):
    def __init__(self, resized_h = 1216, resized_w = 1936):
        super().__init__()
        s = storage.Storage()
        self.__dtc_path = s.dataset_path(os.path.join("edgeai", "dtc"))
        self.__seg_path = s.dataset_path(os.path.join("edgeai", "seg"))
        self.__rgb_path_dict = {}
        self.__json_path_dict = {}
        self.__seg_rgb_path_dict = {}
        self.__seg_lbl_path_dict = {}
        self.__resized_h = resized_h
        self.__resized_w = resized_w
        self.__org_w = 1936
        self.__org_h = 1216

        self.conv_dict = \
        {
            "Car":"car",
            "Bicycle":"bicycle",
            "Pedestrian":"person",
            "Signal":"traffic light",
            "Signs":"traffic sign",
            "Truck":"truck",
            "Bus":"bus",
            "SVehicle":"special",
            "Motorbike":"motorcycle",
            "Train":"train",
        }
        for v in self.conv_dict.values():
            assert(v in self.label_dict.keys())
        
        self.seg_palette = \
        {
            "car"           : [0  ,   0, 255],
            "bus"           : [193, 214,   0],
            "truck"         : [180,   0, 129],
            "special"       : [255, 121, 166],
            "person"        : [255,   0,   0],
            "motorcycle"    : [208, 149,   1],
            "traffic light" : [255, 255,   0],
            "traffic sign"  : [255, 134,   0],
            "sky"           : [  0, 152, 225],
            "building"      : [  0, 203, 151],
            "vegetation"    : [ 85, 255,  50],
            "wall"          : [ 92, 136, 125],
            "lane"          : [ 69,  47, 142],
            "ground"        : [136,  45,  66],
            "sidewalk"      : [  0, 255, 255],
            "road shoulder" : [215,   0, 255],
            "static"        : [180, 131, 135],
            "out of eval"   : [ 81,  99,   0],
            "ego vehicle"   : [ 86,  62,  67],
        }
        for k in self.seg_palette.keys():
            assert(k in self.label_dict.keys())
        
    def __update_list(self, data_type):
        assert((data_type == "train") or \
               (data_type == "test"  ) or \
               (data_type == "debug" ))
        
        max_num = None
        if data_type == "debug":
            max_num = 100
            data_type = "test"
        
        # RGB image
        if not data_type in self.__rgb_path_dict.keys():
            rgb_dir_path = os.path.join(self.__dtc_path, "dtc_{}_images".format(data_type))
            assert(os.path.exists(rgb_dir_path))
            assert(os.path.isdir(rgb_dir_path))
            rgb_path_list = fileio.get_file_list(tgt_dir = rgb_dir_path,
                                                 tgt_ext = ".jpg",
                                                 max_num = max_num)
            rgb_path_list.sort()
            self.__rgb_path_dict[data_type] = rgb_path_list
        
        # JSON
        if not data_type in self.__json_path_dict.keys():
            json_dir_path = os.path.join(self.__dtc_path, "dtc_{}_annotations".format(data_type))
            if os.path.exists(json_dir_path):
                assert(os.path.isdir(json_dir_path))
                json_path_list = fileio.get_file_list(tgt_dir = json_dir_path,
                                                      tgt_ext = ".json",
                                                      max_num = max_num)
                json_path_list.sort()
                self.__json_path_dict[data_type] = json_path_list
                
                # matching check
                assert(len(self.__rgb_path_dict[data_type]) == len(self.__json_path_dict[data_type]))
                for i, json_path in enumerate(self.__json_path_dict[data_type]):
                    json_name = fileio.get_file_name(json_path)
                    json_key = json_name[:-len(".json")]
                    rgb_path = self.__rgb_path_dict[data_type][i]
                    rgb_name = fileio.get_file_name(rgb_path)
                    rgb_key = rgb_name[:-len(".jpg")]
                    assert(rgb_key == json_key)

        return data_type
    
    def __update_seg_list(self, data_type):
        assert((data_type == "train") or \
               (data_type == "test"  ) or \
               (data_type == "debug" ))
        
        max_num = None
        if data_type == "debug":
            max_num = 100
            data_type = "test"
        
        # RGB image
        if not data_type in self.__seg_rgb_path_dict.keys():
            seg_rgb_dir_path = os.path.join(self.__seg_path, "seg_{}_images".format(data_type))
            assert(os.path.exists(seg_rgb_dir_path))
            assert(os.path.isdir(seg_rgb_dir_path))
            seg_rgb_path_list = fileio.get_file_list(tgt_dir = seg_rgb_dir_path,
                                                     tgt_ext = ".jpg",
                                                     max_num = max_num)
            seg_rgb_path_list.sort()
            self.__seg_rgb_path_dict[data_type] = seg_rgb_path_list
        
        # annotation
        if not data_type in self.__seg_lbl_path_dict.keys():
            seg_lbl_dir_path = os.path.join(self.__seg_path, "seg_{}_annotations".format(data_type))
            if os.path.exists(seg_lbl_dir_path):
                assert(os.path.exists(seg_lbl_dir_path))
                assert(os.path.isdir(seg_lbl_dir_path))
                seg_lbl_path_list = fileio.get_file_list(tgt_dir = seg_lbl_dir_path,
                                                         tgt_ext = ".png",
                                                         max_num = max_num)
                seg_lbl_path_list.sort()
                self.__seg_lbl_path_dict[data_type] = seg_lbl_path_list
                
                # matching check
                assert(len(self.__seg_rgb_path_dict[data_type]) == len(self.__seg_lbl_path_dict[data_type]))
                for i, lbl_path in enumerate(self.__seg_lbl_path_dict[data_type]):
                    lbl_name = fileio.get_file_name(lbl_path)
                    lbl_key = lbl_name[:-len(".png")]
                    rgb_path = self.__seg_rgb_path_dict[data_type][i]
                    rgb_name = fileio.get_file_name(rgb_path)
                    rgb_key = rgb_name[:-len(".jpg")]
                    assert(rgb_key == lbl_key)

        return data_type
    
    def get_sample_num(self, data_type):
        data_type = self.__update_list(data_type)
        return len(self.__rgb_path_dict[data_type])
    
    def get_seg_sample_num(self, data_type):
        data_type = self.__update_seg_list(data_type)
        return len(self.__seg_rgb_path_dict[data_type])
    
    def get_vertices_data(self, data_type, tgt_words_list = None, index = None, flip = None):
        data_type = self.__update_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__rgb_path_dict[data_type]))
        if flip is None:
            flip = np.random.randint(2).astype(np.bool)
        
        rect_labels = np.empty(0).astype(np.int)
        rects = np.empty((0, 4)).astype(np.float)
        poly_labels = np.empty(0).astype(np.int)
        polygons = []
        if data_type in self.__json_path_dict.keys():
            json_path = self.__json_path_dict[data_type][index]
            info = json.load(open(json_path))
            obj_list = info["labels"]
            
            for obj in obj_list:
                label_name   = self.conv_dict[obj["category"]]
                if not (label_name in self.label_dict.keys()):
                    print(label_name)
                    assert(0)
                if tgt_words_list is not None:
                    if super().exists_in_words(label_name, tgt_words_list) is False:
                        continue
                assert(label_name in self.label_dict.keys())
                label_value = self.label_dict[label_name]
                if label_value != 0:
                    if "box2d" in obj.keys():
                        box_dict = obj["box2d"]
                        x0 = box_dict["x1"] / self.__org_w
                        x1 = box_dict["x2"] / self.__org_w
                        y0 = box_dict["y1"] / self.__org_h
                        y1 = box_dict["y2"] / self.__org_h
                        assert(x0 <= x1)
                        assert(y0 <= y1)
                        rects = np.append(rects, np.array([y0, x0, y1, x1]).reshape(1, 4), axis = 0)
                        rect_labels = np.append(rect_labels, label_value)
                    elif "poly2d" in obj.keys():
                        polygon = np.array(obj["poly2d"][0]["vertices"]).reshape(-1, 2)
                        if polygon.shape[0] >= 3:
                            polygon = polygon[:,::-1]   # 順番をx,yからy,xに変更
                            polygon = polygon / np.array([self.__org_h, self.__org_w])
                            polygons.append(polygon)
                            poly_labels = np.append(poly_labels, label_value)
        
            rect_labels = super().convert_label_org_val(rect_labels, tgt_words_list)
        
        rgb_path = self.__rgb_path_dict[data_type][index]
        rgb_pil = Image.open(rgb_path)
        rgb_pil = rgb_pil.resize((self.__resized_w, self.__resized_h))
        rgb_arr = np.asarray(rgb_pil)
        
        if flip is True:
            rgb_arr = rgb_arr[:,::-1,:]
            if rects.size > 0:
                rects[:,[1, 3]] = 1.0 - rects[:,[3, 1]]
            if len(polygons) > 0:
                for i in range(len(polygons)):
                    polygons[i][:,1] = 1.0 - polygons[i][:,1]
        if 0: #debug view
            self.summary_vertices_data(rgb_arr, rect_labels, rects, poly_labels, polygons).show()
            exit()
        
        assert(rgb_arr.ndim == 3)
        if rect_labels.size > 0:
            assert(rect_labels.ndim == 1)
        if rects.size > 0:
            assert(rects.ndim == 2)
        return rgb_arr, rect_labels, rects, poly_labels, polygons
    
    def get_seg_data(self, data_type, tgt_words_list, index = None, flip = None):
        self.__update_seg_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__seg_rgb_path_dict[data_type]))
        if flip is None:
            flip = np.random.randint(2).astype(np.bool)
        lbl_path = self.__seg_lbl_path_dict[data_type][index]
        org_lbl_arr = np.asarray(Image.open(lbl_path))
        assert(org_lbl_arr.ndim == 3)
        
        # 当面はデバッグ残す
        if 0:
            for y in range(org_lbl_arr.shape[0]):
                for x in range(org_lbl_arr.shape[1]):
                    if not (list(org_lbl_arr[y, x]) in self.seg_palette.values()):
                        print(list(org_lbl_arr[y, x])) 
                    assert(list(org_lbl_arr[y, x]) in self.seg_palette.values())

        lbl_arr = np.zeros(list(org_lbl_arr.shape)[:-1]).astype(np.int)
        for i, tgt_words in enumerate(tgt_words_list):
            for tgt_word in tgt_words:
                assert(tgt_word in self.label_dict.keys())
                if tgt_word in self.seg_palette.keys():
                    col = self.seg_palette[tgt_word]
                    fill = np.all(org_lbl_arr == col, axis = 2)
                    lbl_arr[fill] = i + 1
        rgb_path = self.__seg_rgb_path_dict[data_type][index]
        rgb_pil = Image.open(rgb_path).resize((self.__resized_w, self.__resized_h))
        rgb_arr = np.asarray(rgb_pil)
        
        if flip is True:
            rgb_arr = rgb_arr[:,::-1,:]
            lbl_arr = lbl_arr[:,::-1]
        return rgb_arr, lbl_arr
    
    def summary_vertices_data(self, rgb_arr, rect_labels, rects, poly_labels, polygons):
        red   = (255,   0,   0, 128)
        blue  = (  0,   0, 255,  64)
        green = (  0, 255,   0,  64)
        white = (255, 255, 255, 255)

        rgb_img = Image.fromarray(rgb_arr)
        dw, dh = rgb_img.size
        draw = ImageDraw.Draw(rgb_img, "RGBA")
        fontheight = draw.getfont().getsize(' ')[1]
        
        if not rects is None:
            for i in range(rects.shape[0]):
                rect = rects[i]
                y0 = (rect[0] * dh)
                x0 = (rect[1] * dw)
                y1 = (rect[2] * dh)
                x1 = (rect[3] * dw)
                draw.rectangle([x0, y0, x1, y1], outline = red)
                for k, v in self.label_dict.items():
                    if v == rect_labels[i]:
                        draw.text([max(0, min(dw - 1, x0)),
                                   max(0, min(dh - 1, y0 - fontheight))],
                                   k,
                                   fill = red)
                        break
        if not polygons is None:
            for i in range(len(polygons)):
                polygon = (polygons[i] * np.array([dh , dw]))[:,::-1]
                for k, v in self.label_dict.items():
                    if v == poly_labels[i]:
                        draw.text([max(0, min(dw - 1, np.average(polygon[:, 0]))),
                                   max(0, min(dh - 1, np.average(polygon[:, 1])) - fontheight)],
                                   k,
                                   fill = white)
                        if k == "lane":
                            fill_col = green
                        elif k == "drivable area":
                            fill_col = blue
                        else:
                            assert(0)
                        draw.polygon(polygon.flatten().tolist(), fill = fill_col)
                        break
        #rgb_img.show()
        return rgb_img
        
    def list_new_category(self):
        data_type = "train"
        json_cat = []
        for json_path in fileio.get_file_list(tgt_dir = os.path.join(self.__data_path, "dtc_{}_annotations".format(data_type)),
                                              name_txt = data_type,
                                              tgt_ext = ".json"):
            info = json.load(open(json_path, "r"))
            for obj in info["labels"]:
                if not obj["category"] in json_cat:
                    json_cat.append(obj["category"])
        for label in json_cat:
            if not label in self.label_dict.keys():
                print(label)
 
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
                        
def make_seg_summary_img(data_type, tgt_words_list):
    e = EdgeAI()
    from tqdm import tqdm
    dst_dir_path = os.path.join(storage.Storage().dataset_path(os.path.join("edgeai", "seg")), "seg_test")

    dst_dir = os.path.join(dst_dir_path, data_type)
    print("writing " + dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i in tqdm(range(e.get_seg_sample_num(data_type))):
        for flip in [False, True]:
            rgb_arr, seg_arr = e.get_seg_data(data_type, tgt_words_list = tgt_words_list, index = i, flip = flip)
            e.summary_seg_data(rgb_arr, seg_arr).save( \
                os.path.join(dst_dir, "{0:06d}".format(i) + "{}.png".format(flip)))

def make_vertices_summary_img(data_type, tgt_labels):
    e = EdgeAI()
    from tqdm import tqdm
    dst_dir_path = os.path.join(storage.Storage().dataset_path(os.path.join("edgeai", "dtc")), "vertices_test")
    
    dst_dir = os.path.join(dst_dir_path, data_type)
    print("writing " + dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for i in tqdm(range(e.get_sample_num(data_type))):
        rgb_arr, rect_labels, rects, poly_labels, polygons = e.get_vertices_data(data_type, index = i, flip = False)
        e.summary_vertices_data(rgb_arr, rect_labels, rects, poly_labels, polygons).save( \
            os.path.join(dst_dir, "{0:06d}.png".format(i)))
    for i in tqdm(range(e.get_sample_num(data_type))):
        rgb_arr, rect_labels, rects, poly_labels, polygons = e.get_vertices_data(data_type, index = i, flip = True)
        e.summary_vertices_data(rgb_arr, rect_labels, rects, poly_labels, polygons).save( \
            os.path.join(dst_dir, "{0:06d}_flip.png".format(i)))

def check_matching(json_dir_path, img_dir_path, ext = ".jpg"):
    assert(os.path.exists(json_dir_path))
    assert(os.path.isdir(json_dir_path))
    assert(os.path.exists(img_dir_path))
    assert(os.path.isdir(img_dir_path))
    for json_path in fileio.get_file_list(json_dir_path):
        json_name = fileio.get_file_name(json_path)
        img_name = json_name.replace(".json", ext)
        img_path = os.path.join(img_dir_path, img_name)
        if not os.path.exists(img_path):
            print(img_path)

def main():
    tgt_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"]]
    #make_vertices_summary_img("train", tgt_words_list)
    make_seg_summary_img("train", tgt_words_list)
if __name__ == "__main__":
    main()
    print("Done.")
