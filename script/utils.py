import numpy as np

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_constant_dim_mask(img):
    mask = []
    for img_ in img:
        img_ = np.reshape(img_, -1)
        mask_ = ~np.all(np.isclose(img_, img_[0]))
        mask.append(mask_)
    return np.array(mask)
