import os
import numpy as np
from PIL import Image

segmented_dir = "./segmented"
main_dir = "./progress"
image_dir = "./images"
source_dir_name = "source"
segment_file_name = "segment.pickle"
progress_file_name = "progress.pickle"


def get_target_image(name):
    path = os.path.join(image_dir, name + ".jpg")
    img = Image.open(path)
    img = np.array(img)
    return img


def get_source_image(index):
    path = os.path.join(image_dir, source_dir_name, "img_%i.jpg" % index)
    pil = Image.open(path)
    img = np.array(pil)
    if not img.shape:
        print "pil: ", pil.size, "np: ", img.shape
        raise IOError("Image %s shape is empty!" % path)
    return img


def get_source_indices():
    img_num_list = []
    for src_img in os.listdir(os.path.join(image_dir, source_dir_name)):
        if "jpg" in src_img:
            img_name = os.path.splitext(src_img)[0]
            img_num = img_name.split("_")[1]
            img_num_list.append(int(img_num))
    return img_num_list


if __name__ == "__main__":
    pass
