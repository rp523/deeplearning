#coding:utf-8
import os
import numpy as np
from PIL import Image
import common.fileio as fio
from tqdm import tqdm

class Dataset:
    
    def __init__(self):
        self.label_dict = {}
        # non-zero values
        self.label_dict["road"         ] = len(self.label_dict.keys()) + 1
        self.label_dict["sidewalk"     ] = len(self.label_dict.keys()) + 1
        self.label_dict["road shoulder"] = len(self.label_dict.keys()) + 1
        self.label_dict["parking"      ] = len(self.label_dict.keys()) + 1
        self.label_dict["rail track"   ] = len(self.label_dict.keys()) + 1
        
        self.label_dict["person"       ] = len(self.label_dict.keys()) + 1
        self.label_dict["rider"        ] = len(self.label_dict.keys()) + 1
        
        self.label_dict["car"          ] = len(self.label_dict.keys()) + 1
        self.label_dict["truck"        ] = len(self.label_dict.keys()) + 1
        self.label_dict["bus"          ] = len(self.label_dict.keys()) + 1
        self.label_dict["on rails"     ] = len(self.label_dict.keys()) + 1
        self.label_dict["bicycle"      ] = len(self.label_dict.keys()) + 1
        self.label_dict["caravan"      ] = len(self.label_dict.keys()) + 1
        self.label_dict["trailer"      ] = len(self.label_dict.keys()) + 1
        self.label_dict["train"        ] = len(self.label_dict.keys()) + 1
        self.label_dict["motorcycle"   ] = len(self.label_dict.keys()) + 1
        self.label_dict["special"      ] = len(self.label_dict.keys()) + 1
        
        self.label_dict["building"     ] = len(self.label_dict.keys()) + 1
        self.label_dict["wall"         ] = len(self.label_dict.keys()) + 1
        self.label_dict["fence"        ] = len(self.label_dict.keys()) + 1
        self.label_dict["guard rail"   ] = len(self.label_dict.keys()) + 1
        self.label_dict["bridge"       ] = len(self.label_dict.keys()) + 1
        self.label_dict["tunnel"       ] = len(self.label_dict.keys()) + 1
        
        self.label_dict["pole"         ] = len(self.label_dict.keys()) + 1
        self.label_dict["traffic sign" ] = len(self.label_dict.keys()) + 1
        self.label_dict["traffic light"] = len(self.label_dict.keys()) + 1
        self.label_dict["billboard"] = len(self.label_dict.keys()) + 1
        
        self.label_dict["vegetation"   ] = len(self.label_dict.keys()) + 1
        self.label_dict["terrain"      ] = len(self.label_dict.keys()) + 1
        
        self.label_dict["sky"          ] = len(self.label_dict.keys()) + 1
        
        self.label_dict["ground"       ] = len(self.label_dict.keys()) + 1
        self.label_dict["dynamic"      ] = len(self.label_dict.keys()) + 1
        self.label_dict["static"       ] = len(self.label_dict.keys()) + 1
        self.label_dict["lane"         ] = len(self.label_dict.keys()) + 1
        self.label_dict["drivable area"] = len(self.label_dict.keys()) + 1
        
        # groups
        self.label_dict["cargroup"     ] = self.label_dict["car"    ] + 128
        self.label_dict["truckgroup"   ] = self.label_dict["truck"  ] + 128
        self.label_dict["busgroup"     ] = self.label_dict["bus"    ] + 128
        self.label_dict["bicyclegroup" ] = self.label_dict["bicycle"] + 128
        self.label_dict["persongroup"  ] = self.label_dict["person" ] + 128
        self.label_dict["pole group"   ] = self.label_dict["pole"   ] + 128
        self.label_dict["ridergroup"   ] = self.label_dict["rider"  ] + 128
        self.label_dict["traingroup"   ] = self.label_dict["train"  ] + 128
        self.label_dict["polegroup"   ]  = self.label_dict["pole"   ] + 128
        #zeros
        self.label_dict["out of roi"          ] = 0
        self.label_dict["rectification border"] = 0
        self.label_dict["ego vehicle"         ] = 0
        self.label_dict["license plate"       ] = 0
        self.label_dict["out of eval"         ] = 0
        # same value
        self.label_dict["bike"] = self.label_dict["motorcycle"]
        self.label_dict["motor"] = self.label_dict["motorcycle"]
        
        self.area_dict = {}
        self.area_dict["driving lane"] = 1
        self.area_dict["beyond line" ] = 2
        
        self.palette = np.array([[255,   0,   0],
                                [0  ,   0, 255],
                                [193, 214,   0],
                                [180,   0, 129],
                                [255, 121, 166],
                                [208, 149,   1],
                                [255, 255,   0],
                                [255, 134,   0],
                                [  0, 152, 225],
                                [  0, 203, 151],
                                [ 85, 255,  50],
                                [ 92, 136, 125],
                                [ 69,  47, 142],
                                [136,  45,  66],
                                [  0, 255, 255],
                                [215,   0, 255],
                                [180, 131, 135],
                                [ 81,  99,   0],
                                [ 86,  62,  67]])
        
    def convert_label_org_val(self, src_labels, words_list = None):
        if words_list is None:
            return src_labels
            
        conv = np.zeros(len(self.label_dict.keys())).astype(np.int)
        for i in range(len(words_list)):
            words = words_list[i]
            for word in words:
                conv[self.label_dict[word]] = i + 1
        dst_labels = src_labels.copy()
        for i in range(src_labels.size):
            dst_labels[i] = conv[src_labels[i]]
        return dst_labels
    
    def exists_in_words(self, label_word, words_list):
        exists = False
        for words in words_list:
            if label_word in words:
                exists = True
                break
        return exists
    
    def convert_to_npy(self, src_dir_path, dst_dir_path, src_ext):
        assert(os.path.exists(src_dir_path))
        assert(os.path.isdir(src_dir_path))
        assert(os.path.exists(dst_dir_path))
        assert(os.path.isdir(dst_dir_path))
        img_path_list = fio.get_file_list(src_dir_path, tgt_ext = src_ext)
        for i in tqdm(range(len(img_path_list))):
            img_path = img_path_list[i]
            dst_path = os.path.join(dst_dir_path, fio.get_file_name(img_path))[:-len(src_ext)] + ".npy"
            np.save(dst_path, np.asarray(Image.open(img_path)).astype(np.uint8))

    def summary_seg_data(self, rgb_arr, lbl_arr):
        alpha = 0.5
        sum_arr = rgb_arr.copy()
        for val in np.unique(lbl_arr).astype(np.int):
            fill_idx = (val == lbl_arr)
            if val != 0:
                ave_col = alpha * rgb_arr[fill_idx] + (1.0 - alpha) * self.palette[val - 1]
            else:
                ave_col = alpha * rgb_arr[fill_idx]
            sum_arr[fill_idx] = ave_col.astype(np.uint8)
        return Image.fromarray(sum_arr)

def main():
    for dat_type in ["train", "val", "test"]:
        src_dir_path = os.path.join(r"/media/isgsktyktt/EC-PHU3/bdd100k/images", dat_type)
        dst_dir_path = os.path.join(r"/media/isgsktyktt/EC-PHU3/bdd100k/npy_images", dat_type)
        if not os.path.exists(dst_dir_path):
            os.makedirs(dst_dir_path)
        Dataset().convert_to_npy(src_dir_path, dst_dir_path, ".jpg")
if "__main__" == __name__:
    main()
    print("Done.")