import random
import numpy as np
import file_handling
import gradient_descent
import image_segment

src_dir = "./source"

if __name__ == "__main__":
    img_name = 'small_butterfly'
    K = 50
    seg_data = image_segment.get_segmented_image(img_name, K)
    labels = seg_data["labels"]
    pixels = seg_data["pixels"]

    img_num_list = file_handling.get_source_indices()
    sorted_patches = image_segment.sort_patch_indices(labels)

    progress = file_handling.get_saved_progress(img_name)
    print "searching..."
    for patch in sorted_patches:
        labelled_indices = np.where(labels == patch)
        patch_pixels = pixels[labelled_indices]
        patch_indices = (labelled_indices[0] - np.min(labelled_indices[0]),
                         labelled_indices[1] - np.min(labelled_indices[1]))

        src_img_index = random.choice(img_num_list)
        env_pixels = file_handling.get_source_image(src_img_index)
        best_src_patch, value = gradient_descent.multi_gradient_descent(env_pixels, patch_pixels, patch_indices, 5)
        seg_data["patch_%i" % patch]["visits"][src_img_index] = (best_src_patch, value)
        file_handling.replace_segment_file(img_name, seg_data)
        file_handling.add_progress(img_name, patch, )
        print "patch %i found" % patch

