#coding: utf-8
import numpy as np

def thinout(src_mat, dst_h, dst_w):
    assert(src_mat.ndim == 2)
    sample_y = (src_mat.shape[0] / dst_h * (np.arange(dst_h) + 0.5) + 0.5).astype(np.int)
    sample_x = (src_mat.shape[1] / dst_w * (np.arange(dst_w) + 0.5) + 0.5).astype(np.int)
    sample_y_mat = sample_y.reshape(-1,  1) + np.zeros((dst_h, dst_w)).astype(np.int)
    sample_x_mat = sample_x.reshape( 1, -1) + np.zeros((dst_h, dst_w)).astype(np.int)
    sample_mat = sample_y_mat * src_mat.shape[0] + sample_x_mat
    out = src_mat.flatten()[sample_mat.flatten()].reshape(dst_h, dst_w)
    return out

if "__main__" == __name__:
    N = 15
    a = np.arange(N**2).reshape(N,N)
    print(a)
    print(thinout(a, 4, 4))
    pass