#coding:utf-8

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
#zeros
label_dict["out of roi"] = 0
label_dict["rectification border"] = 0
label_dict["ego vehicle"  ] = 0
label_dict["license plate"] = 0
# same value
label_dict["bike"] = label_dict["motorcycle"]
label_dict["motor"] = label_dict["motorcycle"]

area_dict = {}
area_dict["driving lane"] = len(area_dict.keys()) + 1
area_dict["beyond line"] = len(area_dict.keys()) + 1
