#coding: utf-8
import numpy as np

def thinout(src_mat, dst_h, dst_w):
    assert(src_mat.ndim == 2)
    sample_y = (src_mat.shape[0] / dst_h * (np.arange(dst_h) + 0.5) + 0.5).astype(np.int32)
    sample_x = (src_mat.shape[1] / dst_w * (np.arange(dst_w) + 0.5) + 0.5).astype(np.int32)
    sample_y_mat = np.zeros(src_mat.shape).astype(np.bool)
    sample_x_mat = np.zeros(src_mat.shape).astype(np.bool)
    sample_y_mat[sample_y,:] = True
    sample_x_mat[:,sample_x] = True
    sample_mat = np.bitwise_and(sample_y_mat, sample_x_mat)
    out = src_mat[sample_mat]
    out = out.reshape(dst_h, dst_w)
    return out

if "__main__" == __name__:
    N = 15
    img_path = r"G:\dataset\cityscapes\leftImg8bit_trainvaltest\leftImg8bit\test\berlin\berlin_000002_000019_leftImg8bit.png"
    from PIL import Image
    a = np.asarray(Image.open(img_path))[:,:,1]
    b = thinout(a, 100, 200)
    Image.fromarray(a).show()
    Image.fromarray(b).show()
    pass