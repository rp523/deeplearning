#coding:utf-8
import sys, os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

sys.path.append("../")
import storage
from common import fileio
from dataset.__init__ import Dataset

class EdgeAIdetection(Dataset):
    def __init__(self, resized_h = 1216, resized_w = 1936):
        super().__init__()
        s = storage.Storage()
        self.__data_path = s.dataset_path("edgeAI_detection")
        self.__rgb_path_dict = {}
        self.__json_path_list = None
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
    
    def __update_list(self, data_type):
        assert((data_type == "train") or \
               (data_type == "test"  ) or \
               (data_type == "debug" ))
        
        max_num = None
        if data_type == "debug":
            max_num = 100
            data_type = "test"
        if not self.__json_path_list:
            json_dir_path = os.path.join(self.__data_path, "dtc_train_annotations")
            assert(json_dir_path)
            self.__json_path_list = fileio.get_file_list(tgt_dir = json_dir_path,
                                                         tgt_ext = ".json",
                                                         max_num = max_num)
        if not data_type in self.__rgb_path_dict.keys():
            rgb_dir_path = os.path.join(self.__data_path, "dtc_{}_images".format(data_type))
            assert(os.path.exists(rgb_dir_path))
            assert(len(self.__json_path_list) > 0)
            
            rgb_path_list = []
            for json_path in self.__json_path_list:
                file_name = fileio.get_file_name(json_path)[:-len(".json")] + ".jpg"
                file_path = os.path.join(rgb_dir_path, file_name)
                if (os.path.exists(file_path)):
                    rgb_path_list.append(file_path)
                else:
                    # still not downloaded
                    pass
            self.__rgb_path_dict[data_type] = rgb_path_list
        return data_type
    
    def get_sample_num(self, data_type):
        data_type = self.__update_list(data_type)
        return len(self.__rgb_path_dict[data_type])
    
    def get_vertices_data(self, data_type, tgt_words_list = None, index = None, flip = None):
        data_type = self.__update_list(data_type)
        if index is None:
            index = np.random.randint(len(self.__json_path_list))
        if index is None:
            flip = np.random.randint(2).astype(np.bool)
        
        json_path = self.__json_path_list[index]

        info = json.load(open(json_path))
        obj_list = info["labels"]
        
        rect_labels = np.empty(0).astype(np.int)
        rects = np.empty((0, 4)).astype(np.float)
        poly_labels = np.empty(0).astype(np.int)
        polygons = []
        
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
            rects[:,[1, 3]] = 1.0 - rects[:,[3, 1]]
            for i in range(len(polygons)):
                polygons[i][:,1] = 1.0 - polygons[i][:,1]
        if 0: #debug view
            self.summary_vertices_data(rgb_arr, rect_labels, rects, poly_labels, polygons).show()
            exit()
        
        return rgb_arr, rect_labels, rects, poly_labels, polygons
    
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
        for bdd in json_cat:
            if not bdd in self.label_dict.keys():
                print(bdd)
 
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
    print(dst_dir)
    for i in tqdm(range(b.get_seg_sample_num(data_type))):
        rgb_arr, seg_arr = b.get_seg_data(data_type, i)
        b.summary_seg_data(rgb_arr, seg_arr).save( \
            os.path.join(dst_dir, "{0:06d}.png".format(i)))

def make_vertices_summary_img(data_type, tgt_labels):
    e = EdgeAIdetection()
    from tqdm import tqdm
    dst_dir_path = os.path.join(storage.Storage().dataset_path("edgeAI_detection"), "vertices_test")
    
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
        
if __name__ == "__main__":
    tgt_words_list = [["car", "truck", "bus", "trailer", "caravan"],
                      ["person", "rider"]]
    make_vertices_summary_img("train", tgt_words_list)
    print("Done.")
