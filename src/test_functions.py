import numpy as np

def bmask(c_out, c_in, i_out, i_in):
    c_out_idx = np.expand_dims(np.arange(c_out) % 3 == i_out, 1)
    c_in_idx = np.expand_dims(np.arange(c_in) % 3 == i_in, 0)
    a1, a2 = np.broadcast_arrays(c_out_idx, c_in_idx)
    return a1, a2