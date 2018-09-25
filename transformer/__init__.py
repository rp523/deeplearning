#coding: utf-8
import os, sys
import numpy as np
sys.path.append("../")
from dataset import label_dict


def transform_seg_label(labels_list, label_map):
    assert(isinstance(label_map, np.ndarray))
    ret = np.zeros(label_map.shape).astype(label_map.dtype)
    label_cnt = 1
    for labels in labels_list:
        for label in labels:
            ret[ret == label_dict[label]] = label_cnt
        label_cnt += 1

def transform_label(labels_list, labels):
    shrink_label = labels.copy()
    label_cnt = 1
    for labels in labels_list:
        for label in labels:
            shrink_label[shrink_label == label_dict[label]] = 1
        label_cnt += 1
    assert((shrink_label != 0).all())
    return shrink_label

if __name__ == "__main__":
    a = np.array([1,1,2,2,0])
    print(a)
    print(a.argsort())
    print(a[a.argsort()])