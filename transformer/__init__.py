#coding: utf-8
import os, sys
import numpy as np
from dataset import Dataset


def transform_seg_label(labels_list, label_map):
    assert(isinstance(label_map, np.ndarray))
    ret = np.zeros(label_map.shape).astype(label_map.dtype)
    label_cnt = 1
    for labels in labels_list:
        for label in labels:
            ret[ret == Dataset().label_dict[label]] = label_cnt
        label_cnt += 1

def transform_label(labels_list, labels):
    shrink_label = labels.copy()
    label_cnt = 1
    for labels in labels_list:
        for label in labels:
            shrink_label[shrink_label == Dataset().label_dict[label]] = 1
        label_cnt += 1
    assert((shrink_label != 0).all())
    return shrink_label

def transform_one_hot(label_vec, class_num):
    assert(label_vec.ndim == 1)
    ret = np.zeros((label_vec.size, class_num)).astype(label_vec.dtype)
    ret[np.arange(ret.shape[0]),label_vec] = 1
    return ret

# anchor情報へ変換
def encode_anchor_label(label_vec, label_rect_mat, anchor, pos_iou_th, neg_iou_th):
    anchor_label_val = -1 * np.ones(anchor.shape[0]).astype(np.int32)
    
    if label_vec.shape[0] <= 0:
        return
    assert(label_vec.size == label_rect_mat.shape[0])
    assert(label_rect_mat.shape[1] == 4)
    assert(anchor.shape[1] == 4)
    reg_label = np.empty(anchor.shape).astype(np.float32)
    cls_label = -1 * np.ones(np.prod(anchor.shape[:2])).astype(np.int32)

    label_h = label_rect_mat[:,2] - label_rect_mat[:,0]
    label_w = label_rect_mat[:,3] - label_rect_mat[:,1]
    label_area = label_h * label_w
    assert((label_area > 0.0).all())
    anchor_h = anchor[:,2] - anchor[:,0]
    anchor_w = anchor[:,3] - anchor[:,1]
    anchor_area = anchor_h * anchor_w
    assert((anchor_area > 0.0).all())
    
    # iouをひととおり計算
    iou_mat = np.zeros((anchor.shape[0], label_vec.size)).astype(np.float32)
    for i in range(label_vec.size):
        label_rect = label_rect_mat[i]
        assert(label_rect.size == 4)
        assert((label_rect[2] - label_rect[0]).all())
        assert((label_rect[3] - label_rect[1]).all())
        # 着目ラベル矩形に対し、アンカー矩形が重なっているかどうか
        is_overlap = np.bitwise_and(np.bitwise_and(label_rect[0] < anchor[:,2], label_rect[2] > anchor[:,0]),
                                    np.bitwise_and(label_rect[1] < anchor[:,3], label_rect[3] > anchor[:,1]))
        if is_overlap.any():
            and_anchor = anchor[is_overlap].copy()
            and_anchor[:,0][and_anchor[:,0] < label_rect[0]] = label_rect[0]
            and_anchor[:,1][and_anchor[:,1] < label_rect[1]] = label_rect[1]
            and_anchor[:,2][and_anchor[:,2] > label_rect[2]] = label_rect[2]
            and_anchor[:,3][and_anchor[:,3] > label_rect[3]] = label_rect[3]
            and_h = (and_anchor[:,2] - and_anchor[:,0])
            and_w = (and_anchor[:,3] - and_anchor[:,1])
            and_area = and_h * and_w
            or_area  = label_area[i] + anchor_area[is_overlap] - and_area
            assert((or_area > 0.0).all())
            iou = and_area / or_area
            assert((iou > 0.0).all())
            iou_mat[is_overlap, i] = iou

    max_iou_label_idx = np.argmax(iou_mat, axis = 1)
    assert(max_iou_label_idx.shape[0] == anchor.shape[0])
    is_pos_anchor = np.max(iou_mat, axis = 1) > pos_iou_th
    is_neg_anchor = np.max(iou_mat, axis = 1) < neg_iou_th
    assert(is_pos_anchor.ndim == 1)
    assert(is_neg_anchor.ndim == 1)
    assert(is_pos_anchor.size == anchor.shape[0])
    assert(is_neg_anchor.size == anchor.shape[0])
    label_table = label_vec + np.zeros((anchor.shape[0], label_vec.size)).astype(np.int32)
    anchor_label_val[is_pos_anchor] = label_table[np.arange(label_table.shape[0]), max_iou_label_idx][is_pos_anchor]
    anchor_label_val[is_neg_anchor] = 0
    
    ret_anchor = np.zeros((anchor_label_val.size, label_rect_mat.shape[1])).astype(np.float32)
    for i in range(anchor_label_val.size):
        if is_pos_anchor[i]:
            ret_anchor[i] = label_rect_mat[max_iou_label_idx[i]]
    assert(ret_anchor.shape[0] == anchor_label_val.size)
    assert(ret_anchor.shape[1] == 4)
    return anchor_label_val, ret_anchor


def make_anchor(anchor_div, size_list = [1.0], asp_list = [0.5, 1.0, 2.0]):
    # まずasp=1.0, size=1.0で作る
    anchor_div_y = anchor_div[0]
    anchor_div_x = anchor_div[1]
    y = np.linspace(0.0, 1.0, anchor_div_y + 1)
    x = np.linspace(0.0, 1.0, anchor_div_x + 1)
    y0 = y[:-1]
    y1 = y[1 :]
    x0 = x[:-1]
    x1 = x[1 :]
    base_rect_mat = np.empty((anchor_div_y, anchor_div_x, 4)).astype(np.float32)
    base_rect_mat[:,:,0] = y0.reshape(-1, 1)
    base_rect_mat[:,:,1] = x0.reshape( 1,-1)
    base_rect_mat[:,:,2] = y1.reshape(-1, 1)
    base_rect_mat[:,:,3] = x1.reshape( 1,-1)
    base_rect_mat = base_rect_mat.reshape(-1, 4)
    assert(base_rect_mat.ndim == 2)
    assert(base_rect_mat.shape[1] == 4)
    center = base_rect_mat.reshape(-1, 2, 2).mean(axis = -2)
            
    anchors = np.empty((len(size_list), len(asp_list), anchor_div_y * anchor_div_x, 4)).astype(np.float32)
    for s in range(len(size_list)):
        for a in range(len(asp_list)):
            h = np.abs(base_rect_mat[:,2] - base_rect_mat[:,0]) * size_list[s] / np.sqrt(asp_list[a])
            w = np.abs(base_rect_mat[:,3] - base_rect_mat[:,1]) * size_list[s] * np.sqrt(asp_list[a])
            anchors[s, a, :, 0] = center[:,0] - 0.5 * h 
            anchors[s, a, :, 1] = center[:,1] - 0.5 * w 
            anchors[s, a, :, 2] = center[:,0] + 0.5 * h 
            anchors[s, a, :, 3] = center[:,1] + 0.5 * w
    anchors[anchors <= 0.0] = 0.0
    anchors[anchors >= 1.0] = 1.0
    assert((anchors <= 1.0).all())
    assert((anchors >= 0.0).all())
    assert(anchors.ndim == 4)
    assert(anchors.shape[3] == 4)
    return anchors.reshape(-1, 4)

def make_anchor_test():
    img_h, img_w = 900, 1500
    img_arr = np.zeros((img_h, img_w)).astype(np.uint8)
    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_img)
    size_list = [2.0**0.0, 2.0**0.25, 2.0**0.50, 2.0**0.75]
    asp_list = [0.5, 1.0, 1.5]
    div_y = 10
    div_x = 10
    anchor = make_anchor([div_y, div_x], size_list, asp_list)
    print(anchor.shape)
    n = 0
    for rect in anchor:
        assert(rect.size == 4)
        y0 = rect[0] * img_h
        x0 = rect[1] * img_w
        y1 = rect[2] * img_h
        x1 = rect[3] * img_w
        draw.rectangle((x0, y0, x1, y1),
                       #fill = 0,
                       outline = 255)
        #draw.text(((x0+x1)/2, (y0+y1)/2), text = str(n))
    pil_img.show()
    
def encode_anchor_label_test():
    from PIL import Image, ImageDraw
    sys.path.append("../")
    from dataset import Dataset
    from dataset.bdd100k import BDD100k
    
    size_list = []
    for val in np.linspace(-1.0, 2.0, 5):
        size_list.append(2.0 ** val)
    asp_list = []
    for val in np.linspace(0.5, 2.0, 5):
        asp_list.append(val)
    div_y = 16
    div_x = 16
    anchor = make_anchor([div_y, div_x], size_list, asp_list)
    bdd = BDD100k()
    for n in range(5000):
        rgb_arr, rect_labels, rects, poly_labels, polygons = \
        bdd.get_vertices_data("val", [["car", "truck", "bus", "trailer", "caravan"]], index = n)
        if 0: # anot data view
            pil_img = Image.fromarray(rgb_arr.astype(np.uint8))
            w, h = pil_img.size
            draw = ImageDraw.Draw(pil_img)
            for rect in rects:
                y0 = rect[0] * h
                x0 = rect[1] * w
                y1 = rect[2] * h
                x1 = rect[3] * w
                draw.rectangle((x0, y0, x1, y1), outline=(255,0,0))
            pil_img.show()
            exit()
            
        if rect_labels.shape[0] > 0:
            valid_label_vec = \
            encode_anchor_label(label_vec = rect_labels,
                                label_rect_mat = rects,
                                anchor = anchor,
                                pos_iou_th = 0.5,
                                neg_iou_th = 0.4)[1]
            if (valid_label_vec > 0).any():
                pil_img = Image.fromarray(rgb_arr.astype(np.uint8))
                w, h = pil_img.size
                draw = ImageDraw.Draw(pil_img)
                for i in range(valid_label_vec.size):
                    label = valid_label_vec[i]
                    col = None
                    if label < 0:
                        pass
                        #col = (0,255,0)
                    elif label == 0:
                        pass
                        #col = (0,0,255)
                    elif label > 0:
                        col = (255,0,0)
                    if col is not None:
                        rect = anchor[i]
                        y0 = rect[0] * h
                        x0 = rect[1] * w
                        y1 = rect[2] * h
                        x1 = rect[3] * w
                        draw.rectangle((x0, y0, x1, y1), outline = col)
                pil_img.show()
                exit()
        print("no hit " + str(n))
if __name__ == "__main__":
    encode_anchor_label_test()