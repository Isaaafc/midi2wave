import numpy as np

def quantize(images, levels=256):
    return (np.digitize(images, np.arange(levels) / levels) - 1).astype('i')
