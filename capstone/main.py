import random
import numpy as np
import file_handling
import gradient_descent
import image_segment

src_dir = "./source"

if __name__ == "__main__":
    img_name = 'butterfly'
    K = 150
    seg_data = image_segment.get_segmented_image(img_name, K)
    labels = seg_data["labels"]
    pixels = seg_data["pixels"]

    img_num_list = file_handling.get_source_indices()

    max_label = np.max(labels)
    hist = np.histogram(labels, bins=max_label)
    order = np.flip(np.argsort(hist[0]), 0)

    for label in order:
        labelled_indices = np.where(labels == label)
        patch_pixels = pixels[labelled_indices]
        patch_indices = (labelled_indices[0] - np.min(labelled_indices[0]),
                         labelled_indices[1] - np.min(labelled_indices[1]))

        src_img_index = random.choice(img_num_list)
        env_pixels = file_handling.get_source_image(src_img_index)
        best_src_patch, value = gradient_descent.multi_gradient_descent(env_pixels, patch_pixels, patch_indices, 5)
        seg_data["patch_%i" % label]["visits"][src_img_index] = (best_src_patch, value)

    file_handling.write_seg_file(seg_data)
