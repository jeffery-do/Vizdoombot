import skimage.color, skimage.transform
import numpy as np

def preprocess(img, resolution):
    new_img = skimage.transform.resize(img, resolution)
    new_img = new_img.astype(np.float32)
    new_img = np.reshape(new_img, (new_img.shape[0], new_img.shape[1], new_img.shape[2]))
    return new_img