#coding:utf-8
import sys, os, subprocess
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    sys.path.append(subprocess.getoutput("pwd"))
import storage
from common import fileio
from dataset.__init__ import Dataset

bdd_seg_dict = {}
bdd_seg_dict["road"] = 0
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

class BDD100k(Dataset):
    
    def __init__(self, resized_h = 720, resized_w = 1280):
        super().__init__()
        s = storage.Storage()
        self.__data_path = s.dataset_path("bdd100k")
        self.__rgb_path_dict = {}
        self.__json_path_dict = {}
        self.__seg_path_dict = {}
        self.__segimg_path_dict = {}
        self.__rgb_h, self.__rgb_w = 720, 1280
        self.__resized_h = resized_h
        self.__resized_w = resized_w

    def __update_seg_list(self, data_type):
        assert((data_type == "train") or \
               (data_type == "val"  ) or \
               (data_type == "test" ))
        if not data_type in self.__seg_path_dict.keys():
            seg_dir_path = os.path.join(self.__data_path, "seg", "labels", data_type)
            if os.path.exists(seg_dir_path):
                seg_path_list_ = []
                for seg_path in fileio.get_file_list(tgt_dir = seg_dir_path,
                                                     tgt_ext = ".png"):
                    seg_path_list_.append(seg_path)
            segimg_dir_path = os.path.join(self.__data_path, "seg", "images", data_type)
            if os.path.exists(segimg_dir_path):
                segimg_path_list_ = []
                for segimg_path in fileio.get_file_list(tgt_dir = segimg_dir_path,
                                                        tgt_ext = ".jpg"):
                    segimg_path_list_.append(segimg_path)
            
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
            for s, seg_key in enumerate(seg_key_list):
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
               (data_type == "test"  ) or \
               (data_type == "debug" ))
        
        max_num = None
        if data_type == "debug":
            max_num = 100
            data_type = "val"
        if not data_type in self.__json_path_dict.keys():
            json_dir_path = os.path.join(self.__data_path, "labels", data_type)
            if os.path.exists(json_dir_path):
                json_path_list_ = []
                for json_path in fileio.get_file_list(tgt_dir = json_dir_path,
                                                      tgt_ext = ".json",
                                                      max_num = max_num):
                    json_path_list_.append(json_path)
            rgb_dir_path = os.path.join(self.__data_path, "images", data_type)
            if os.path.exists(rgb_dir_path):
                rgb_path_list_ = []
                for rgb_path in fileio.get_file_list(tgt_dir = rgb_dir_path,
                                                     tgt_ext = ".jpg"):
                    rgb_path_list_.append(rgb_path)
            assert(len(json_path_list_) > 0)
            assert(len(rgb_path_list_) > 0)

            json_key_list = []
            rgb_key_list  = []
            for json_path in json_path_list_:
                key = fileio.get_file_name(json_path)[:-len(".json")]
                json_key_list.append(key)
            for rgb_path in rgb_path_list_:
                key = fileio.get_file_name(rgb_path)[:-len(".jpg")]
                rgb_key_list.append(key)
            assert(len(json_key_list) > 0)
            assert(len(rgb_key_list) > 0)
            
            json_path_list = []
            rgb_path_list  = []
            for j in range(len(json_key_list)):
                json_key = json_key_list[j]
                if json_key in rgb_key_list:
                    json_path_list.append(json_path_list_[j])
                    i = rgb_key_list.index(json_key)
                    rgb_path_list.append(rgb_path_list_[i])
            assert(len(json_path_list) == len(rgb_path_list))
            assert(len(json_path_list) > 0)
            assert(len(rgb_path_list) > 0)
            self.__rgb_path_dict[data_type]  = rgb_path_list
            self.__json_path_dict[data_type] = json_path_list
        return data_type
    
    def get_sample_num(self, data_type):
        data_type = self.__update_list(data_type)
        return len(self.__rgb_path_dict[data_type])
    
    def get_seg_sample_num(self, data_type):
        self.__update_seg_list(data_type)
        return len(self.__seg_path_dict[data_type])
    
    def get_seg_data(self, data_type, tgt_words_list = None, index = None, flip = None):
        self.__update_seg_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__seg_path_dict[data_type]))
        seg_path = self.__seg_path_dict[data_type][index]
        seg_arr = np.asarray(Image.open(seg_path))
        
        lbl_arr = np.zeros(seg_arr.shape).astype(np.uint8)
        for i, tgt_words in enumerate(tgt_words_list):
            for tgt_word in tgt_words:
                if tgt_word in bdd_seg_dict.keys():
                    tgt_seg_val = bdd_seg_dict[tgt_word]
                    fill_idx = seg_arr == tgt_seg_val
                    if fill_idx.any():
                        lbl_arr[fill_idx] = i + 1
        rgb_path = self.__segimg_path_dict[data_type][index]
        rgb_arr = np.asarray(Image.open(rgb_path))

        if flip is None:
            flip = np.bool(np.random.randint(2))
        if flip is True:
            rgb_arr = rgb_arr[:,::-1,:]
            lbl_arr = lbl_arr[:,::-1]
        return rgb_arr, lbl_arr
    
    def get_vertices_data(self, data_type, tgt_words_list = None, index = None, flip = None):
        data_type = self.__update_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__json_path_dict[data_type]))
        if flip is None:
            flip = np.bool(np.random.randint(2))
        
        json_path = self.__json_path_dict[data_type][index]

        info = json.load(open(json_path))
        obj_list = info["labels"]
        
        rect_labels = np.empty(0).astype(np.int)
        rects = np.empty((0, 4)).astype(np.float)
        poly_labels = np.empty(0).astype(np.int)
        polygons = []
        
        for obj in obj_list:
            label_name   = obj["category"]
            if not (label_name in self.label_dict.keys()):
                print(label_name)
            if tgt_words_list is not None:
                if super().exists_in_words(label_name, tgt_words_list) is False:
                    continue
            assert(label_name in self.label_dict.keys())
            label_value = self.label_dict[label_name]
            if label_value != 0:
                if "box2d" in obj.keys():
                    box_dict = obj["box2d"]
                    x0 = box_dict["x1"] / self.__rgb_w
                    x1 = box_dict["x2"] / self.__rgb_w
                    y0 = box_dict["y1"] / self.__rgb_h
                    y1 = box_dict["y2"] / self.__rgb_h
                    assert(x0 <= x1)
                    assert(y0 <= y1)
                    rects = np.append(rects, np.array([y0, x0, y1, x1]).reshape(1, 4), axis = 0)
                    rect_labels = np.append(rect_labels, label_value)
                elif "poly2d" in obj.keys():
                    polygon = np.array(obj["poly2d"][0]["vertices"]).reshape(-1, 2)
                    if polygon.shape[0] >= 3:
                        polygon = polygon[:,::-1]   # 順番をx,yからy,xに変更
                        polygon = polygon / np.array([self.__rgb_h, self.__rgb_w])
                        polygons.append(polygon)
                        poly_labels = np.append(poly_labels, label_value)
    
        rect_labels = super().convert_label_org_val(rect_labels, tgt_words_list)
        
        rgb_path = self.__rgb_path_dict[data_type][index]
        rgb_pil = Image.open(rgb_path)
        rgb_pil = rgb_pil.resize((self.__resized_w, self.__resized_h))
        rgb_arr = np.asarray(rgb_pil)
        
        if flip is True:
            rgb_arr = rgb_arr[:,::-1,:]
            rects[:,[1, 3]] = 1.0 - rects[:,[3, 1]]
            for i in range(len(polygons)):
                polygons[i][:,1] = 1.0 - polygons[i][:,1]
        if 0: #debug view
            self.summary_vertices_data(rgb_arr, rect_labels, rects, poly_labels, polygons).show()
            exit()
        
        return rgb_arr, rect_labels, rects, poly_labels, polygons
    
    def __conv_vertices_to_drivable_edge(self, polygons, poly_labels, h_pix, w_pix):
        fill_arr = np.zeros((h_pix, w_pix)).astype(np.uint8)
        fill_pil = Image.fromarray(fill_arr)
        draw = ImageDraw.Draw(fill_pil)
        # いったん興味領域を白塗りする
        for i, poly_label in enumerate(poly_labels):
            if poly_label == self.label_dict["drivable area"]:
                polygon_pix = polygons[i] * np.array([h_pix, w_pix])
                polygon_pix = np.array(polygon_pix[:,::-1] + 0.5).astype(np.int)
                draw.polygon(polygon_pix.flatten().tolist(), fill = 255)
        fill_arr = np.asarray(fill_pil)
        valid_row = np.any(fill_arr > 0, axis = 1)
        tgt_arr = fill_arr[valid_row]
        # 塗られた領域の左端、右端をエッジとする
        left_edge_x  = -1 * np.ones(h_pix).astype(np.float)
        right_edge_x = -1 * np.ones(h_pix).astype(np.float)
        is_valid = (tgt_arr > 0)
        if is_valid.any():
            idx_cand = np.array(np.where(is_valid))
            is_boundary_idx = (idx_cand[0][1:] - idx_cand[0][:-1] > 0)
    
            is_min_idx = np.append([True], is_boundary_idx)
            left_edge_xpix_vec = idx_cand[1][is_min_idx]
            left_edge_x[valid_row] = left_edge_xpix_vec / w_pix
            
            is_max_idx = np.append(is_boundary_idx, [True])
            right_edge_xpix_vec = idx_cand[1][is_max_idx]
            right_edge_x[valid_row] = 1.0 - right_edge_xpix_vec / w_pix

        return np.append( left_edge_x.reshape(-1, 1),
                         right_edge_x.reshape(-1, 1),
                         axis = 1)
        
    def get_drivable_edge_data(self, data_type, index = None, flip = None, h_pix = None, w_pix = None):
        rgb_arr, rect_labels, rects, poly_labels, polygons = self.get_vertices_data(data_type, index = index, flip = flip)
        drivable_edge = self.__conv_vertices_to_drivable_edge(polygons, poly_labels, rgb_arr.shape[0], rgb_arr.shape[1])
        if (h_pix != None) and (w_pix != None):
            rgb_arr = np.asarray(Image.fromarray(rgb_arr).resize((w_pix, h_pix)))
            org_h = drivable_edge.shape[0]
            assert(h_pix <= org_h)
            mask = np.linspace(0, org_h - 1, h_pix).astype(np.int)
            drivable_edge = drivable_edge[mask]
        return rgb_arr, drivable_edge

    def summary_vertices_data(self, rgb_arr, rect_labels, rects, poly_labels, polygons):
        red   = (255,   0,   0, 128)
        blue  = (  0,   0, 255,  64)
        green = (  0, 255,   0,  64)
        white = (255, 255, 255, 255)

        rgb_img = Image.fromarray(rgb_arr)
        dw, dh = rgb_img.size
        draw = ImageDraw.Draw(rgb_img, "RGBA")
        fontheight = draw.getfont().getsize(' ')[1]
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
    
    def summary_drivable_edge_data(self, rgb_arr, drivable_edge, valid_range_start = None, valid_range_end = None):
        is_edge_exist_row = drivable_edge[:,0] >= 0
        assert((is_edge_exist_row == (drivable_edge[:,1] >= 0)).all())
        assert(drivable_edge.shape[0] == rgb_arr.shape[0])
        draw_arr = rgb_arr.copy()
        draw_pil = Image.fromarray(draw_arr.astype(np.uint8))
        draw = ImageDraw.Draw(draw_pil)
        col = (255, 0, 0)
        for i in range(rgb_arr.shape[0] - 1):
            y0 = i
            y1 = i + 1
            if (valid_range_start != None) and (valid_range_end != None):
                if (y0 < valid_range_start) or (y1 > valid_range_end):
                    col = (0, 0, 0,)
            for j in range(2):
                x0 = int(drivable_edge[i    ][j] * rgb_arr.shape[1])
                x1 = int(drivable_edge[i + 1][j] * rgb_arr.shape[1])
                if (x0 >= 0) and (x1 >= 0):
                    if j == 1:
                        x0 = rgb_arr.shape[1] - 1 - x0
                        x1 = rgb_arr.shape[1] - 1 - x1
                    draw.line(((x0, y0), (x1, y1)), fill = col, width = 0)
        return draw_pil
        
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
            if not bdd in self.label_dict.keys():
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
                        
def make_seg_summary_img(data_type, tgt_words_list):
    b = BDD100k()
    from tqdm import tqdm
    dst_dir_path = os.path.join(storage.Storage().dataset_path("bdd100k"), "seg_test")

    dst_dir = os.path.join(dst_dir_path, data_type)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    print(dst_dir)
    for i in tqdm(range(b.get_seg_sample_num(data_type))):
        for flip in [False]:
            rgb_arr, seg_arr = b.get_seg_data(data_type = data_type, tgt_words_list = tgt_words_list, index = i, flip = flip)
            b.summary_seg_data(rgb_arr, seg_arr).save( \
                os.path.join(dst_dir, "{0:06d}".format(i) + "{}.png".format(flip)))

def make_vertices_summary_img(data_type, tgt_labels):
    b = BDD100k()
    from tqdm import tqdm
    dst_dir_path = os.path.join(storage.Storage().dataset_path("bdd100k"), "vertices_test")
    
    dst_dir = os.path.join(dst_dir_path, data_type)
    print("writing " + dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for i in tqdm(range(b.get_sample_num(data_type))):
        rgb_arr, rect_labels, rects, poly_labels, polygons = b.get_vertices_data(data_type, index = i, flip = False)
        b.summary_vertices_data(rgb_arr, rect_labels, rects, poly_labels, polygons).save( \
            os.path.join(dst_dir, "{0:06d}.png".format(i)))
    for i in tqdm(range(b.get_sample_num(data_type))):
        rgb_arr, rect_labels, rects, poly_labels, polygons = b.get_vertices_data(data_type, index = i, flip = True)
        b.summary_vertices_data(rgb_arr, rect_labels, rects, poly_labels, polygons).save( \
            os.path.join(dst_dir, "{0:06d}_flip.png".format(i)))
        
def make_drivablearea_summary_img(data_type):
    b = BDD100k()
    from tqdm import tqdm
    dst_dir_path = os.path.join(storage.Storage().dataset_path("bdd100k"), "drivable_edge_test")
    
    dst_dir = os.path.join(dst_dir_path, data_type)
    print("writing " + dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for flip in [False, True]:
        for i in tqdm(range(b.get_sample_num(data_type))):
            rgb_arr, drivable_edge = b.get_drivable_edge_data(data_type, index = i, flip = flip, h_pix = 400, w_pix = 800)
            b.summary_drivable_edge_data(rgb_arr, drivable_edge).save( \
                os.path.join(dst_dir, "{0:06d}".format(i) + "{}.png".format(flip)))
        
if __name__ == "__main__":
    dtc_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"]]
    seg_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"],
                      ["road"]]
    make_drivablearea_summary_img("train")
    #make_vertices_summary_img("train", dtc_words_list)
    #make_seg_summary_img("train", seg_words_list)
    print("Done.")
