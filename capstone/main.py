import image_segment
from gd_env import Env, GD_Searcher
import numpy as np
import os
import time
import random
from scipy import ndimage, misc

src_dir = "./src"

if __name__ == "__main__":
    img_name = 'butterfly'
    K = 150
    seg_data = image_segment.get_segmented_img(img_name, K)
    labels = seg_data["labels"]
    pixels = seg_data["pixels"]

    img_num_list = []
    for src_img in os.listdir(src_dir):
        if "jpg" in src_img:            
            img_name = os.path.splitext(src_img)[0]
            img_num = img_name.split("_")[1]
            img_num_list.append(int(img_num))

    max_label = np.max(labels)
    hist = np.histogram(labels, bins = max_label)
    order = np.flip(np.argsort(hist[0]), 0)
    
    for label in order:        
        labelled_indices = np.where(labels == label)
        patch_pixels = pixels[labelled_indices]    
        patch_indices = (patch_indices[0] - np.min(patch_indices[0]),
                             patch_indices[1] - np.min(patch_indices[1]))
        
        src_img_index = random.choice(img_num_list)
        env_pixels = ndimage.imread("./src/img_%i.jpg" % src_img_index)            
        best_src_patch, value = gradient_descent.grad_descent(env_pixels, patch_pixels, patch_indices)
        patchProgressFile = get_file_for_patch(label)

        #pseudocode
        if src_img_index in patchProgressFile:
            del(src_img_index)
        patchProgressFile.writeline(src_img_index, best_src_patch, value)
        
        print patch_index
