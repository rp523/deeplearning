#coding:utf-8
import numpy as np
label_dict = {}

# non-zero values
label_dict["road"         ] = len(label_dict.keys()) + 1
label_dict["sidewalk"     ] = len(label_dict.keys()) + 1
label_dict["parking"      ] = len(label_dict.keys()) + 1
label_dict["rail track"   ] = len(label_dict.keys()) + 1

label_dict["person"       ] = len(label_dict.keys()) + 1
label_dict["rider"        ] = len(label_dict.keys()) + 1

label_dict["car"          ] = len(label_dict.keys()) + 1
label_dict["truck"        ] = len(label_dict.keys()) + 1
label_dict["bus"          ] = len(label_dict.keys()) + 1
label_dict["on rails"     ] = len(label_dict.keys()) + 1
label_dict["bicycle"      ] = len(label_dict.keys()) + 1
label_dict["caravan"      ] = len(label_dict.keys()) + 1
label_dict["trailer"      ] = len(label_dict.keys()) + 1
label_dict["train"        ] = len(label_dict.keys()) + 1
label_dict["motorcycle"   ] = len(label_dict.keys()) + 1

label_dict["building"     ] = len(label_dict.keys()) + 1
label_dict["wall"         ] = len(label_dict.keys()) + 1
label_dict["fence"        ] = len(label_dict.keys()) + 1
label_dict["guard rail"   ] = len(label_dict.keys()) + 1
label_dict["bridge"       ] = len(label_dict.keys()) + 1
label_dict["tunnel"       ] = len(label_dict.keys()) + 1

label_dict["pole"         ] = len(label_dict.keys()) + 1
label_dict["traffic sign" ] = len(label_dict.keys()) + 1
label_dict["traffic light"] = len(label_dict.keys()) + 1
label_dict["billboard"] = len(label_dict.keys()) + 1

label_dict["vegetation"   ] = len(label_dict.keys()) + 1
label_dict["terrain"      ] = len(label_dict.keys()) + 1

label_dict["sky"          ] = len(label_dict.keys()) + 1

label_dict["ground"       ] = len(label_dict.keys()) + 1
label_dict["dynamic"      ] = len(label_dict.keys()) + 1
label_dict["static"       ] = len(label_dict.keys()) + 1
label_dict["lane"         ] = len(label_dict.keys()) + 1
label_dict["drivable area"] = len(label_dict.keys()) + 1

# groups
label_dict["cargroup"     ] = label_dict["car"    ] + 128
label_dict["truckgroup"   ] = label_dict["truck"  ] + 128
label_dict["busgroup"     ] = label_dict["bus"    ] + 128
label_dict["bicyclegroup" ] = label_dict["bicycle"] + 128
label_dict["persongroup"  ] = label_dict["person" ] + 128
label_dict["pole group"   ] = label_dict["pole"   ] + 128
label_dict["ridergroup"   ] = label_dict["rider"  ] + 128
label_dict["traingroup"   ] = label_dict["train"  ] + 128
label_dict["polegroup"   ]  = label_dict["pole"   ] + 128
#zeros
label_dict["out of roi"          ] = 0
label_dict["rectification border"] = 0
label_dict["ego vehicle"         ] = 0
label_dict["license plate"       ] = 0
label_dict["out of eval"         ] = 0
# same value
label_dict["bike"] = label_dict["motorcycle"]
label_dict["motor"] = label_dict["motorcycle"]

area_dict = {}
area_dict["driving lane"] = 1
area_dict["beyond line" ] = 2

def convert_label_org_val(src_labels, words_list):
    conv = np.zeros(len(label_dict.keys())).astype(np.int)
    
    for i in range(len(words_list)):
        words = words_list[i]
        for word in words:
            conv[label_dict[word]] = i + 1
    
    dst_labels = src_labels.copy()
    for i in range(src_labels.size):
        dst_labels[i] = conv[src_labels[i]]
    
    return dst_labels
