import numpy as np
import pandas as pd
from nilearn import datasets, image

def find_region_indices(labels, patterns):
    """Return list of indices for labels containing any of the case-insensitive patterns."""
    hits = []
    for i, lab in enumerate(labels):
        lab_l = lab.lower()
        if any(p.lower() in lab_l for p in patterns):
            hits.append(i)
    return hits

def mask_from_labels(label_img, labels_list, substrings):
    """Return a binary mask selecting any labels containing given substrings (case-insensitive)."""
    lbl_map = np.zeros_like(label_img.get_fdata(), dtype=bool)
    # labels_list[0] is usually 'Background'
    for idx, name in enumerate(labels_list):
        if idx == 0 or name is None:
            continue
        if any(s.lower() in name.lower() for s in substrings):
            lbl_map |= (label_img.get_fdata() == idx)
    return image.new_img_like(label_img, lbl_map.astype("uint8"))

