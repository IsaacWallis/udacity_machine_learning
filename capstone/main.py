import random
import numpy as np
import file_handling
import gradient_descent
import image_segment
import sql_model

src_dir = "./source"

if __name__ == "__main__":
    img_name = 'small_butterfly'
    K = 50
    target_image = sql_model.get_target_image(img_name, K)
    labels = target_image.labels
    pixels = target_image.pixels

    img_num_list = file_handling.get_source_indices()
    sorted_patches = image_segment.sort_patch_indices(labels)

    print "searching..."
    count = 0
    for patch in sorted_patches:
        labelled_indices = np.where(labels == patch)
        patch_pixels = pixels[labelled_indices]
        patch_indices = (labelled_indices[0] - np.min(labelled_indices[0]),
                         labelled_indices[1] - np.min(labelled_indices[1]))

        src_img_index = random.choice(img_num_list)

        env_pixels = file_handling.get_source_image(src_img_index)

        best_src_patch, value = gradient_descent.multi_gradient_descent(env_pixels, patch_pixels, patch_indices, 5)

        visited_image = sql_model.SourceImage(id=src_img_index)
        best_state = sql_model.State(source=visited_image, translation_x=best_src_patch[0],
                                     translation_y=best_src_patch[1])
        searching_patch = sql_model.TargetPatch(id=patch)
        searching_patch.visits.append(best_state)
        target_image.segments.append(searching_patch)
        sql_model.get_session().commit()

        print "%i: patch %i found" % (count, patch)
        count += 1
