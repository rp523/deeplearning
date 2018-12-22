#coding: utf-8
import numpy as np

def thinout(src_mat, dst_h, dst_w):
    assert(src_mat.ndim == 2)
    sample_y = (np.linspace(0, src_mat.shape[0] - 1, dst_h) + 0.5).astype(np.int)
    sample_x = (np.linspace(0, src_mat.shape[1] - 1, dst_w) + 0.5).astype(np.int)
    sample_y_mat = sample_y.reshape(-1,  1) + np.zeros((dst_h, dst_w)).astype(np.int)
    sample_x_mat = sample_x.reshape( 1, -1) + np.zeros((dst_h, dst_w)).astype(np.int)
    sample_mat = sample_y_mat * dst_w + sample_x_mat
    out = src_mat.flatten()[sample_mat.flatten()].reshape(dst_h, dst_w)
    return out

if "__main__" == __name__:
    N = 10
    a = np.arange(N**2).reshape(N,N)
    print(a)
    print(thinout(a, 4, 4))
    pass