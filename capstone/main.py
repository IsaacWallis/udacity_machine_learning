import random
import numpy as np
import file_handling
import gradient_descent
import image_segment
import sql_model

src_dir = "./source"

if __name__ == "__main__":
    img_name = 'australian_butterfly'
    K = 150
    target_image = sql_model.get_target_image(img_name, K)
    target_labels = target_image.labels
    sorted_patches = image_segment.sort_patch_indices(target_labels)
    target_pixels = target_image.pixels

    img_num_list = file_handling.get_source_indices()

    print "searching..."
    count = 0
    for i in range(10):
        for patch in sorted_patches:
            labelled_indices = np.where(target_labels == patch)
            patch_pixels = target_pixels[labelled_indices]
            patch_indices = (labelled_indices[0] - np.min(labelled_indices[0]),
                             labelled_indices[1] - np.min(labelled_indices[1]))

            src_img_index = random.choice(img_num_list)
            env_pixels = file_handling.get_source_image(src_img_index)
            while env_pixels.shape[0] <= np.max(patch_indices[0]) or env_pixels.shape[1] <= np.max(patch_indices[1]):
                print "source image too small, choosing another"
                src_img_index = random.choice(img_num_list)
                env_pixels = file_handling.get_source_image(src_img_index)

            best_src_patch, value = gradient_descent.multi_gradient_descent(env_pixels, patch_pixels, patch_indices, 5)

            visited_image = sql_model.SourceImage(id=src_img_index)

            searching_patch = sql_model.get_session(img_name).query(sql_model.TargetPatch) \
                .filter_by(parent_name=img_name,id=patch).first()
            if not searching_patch:
                searching_patch = sql_model.TargetPatch(id=patch)
                target_image.segments.append(searching_patch)

            best_state = sql_model.State(source=visited_image.id,
                                         x=int(round(best_src_patch[0])),
                                         y=int(round(best_src_patch[1])),
                                         target_patch=searching_patch.id,
                                         loss=value
                                         )

            sql_model.get_session(img_name).add(best_state)
            sql_model.get_session(img_name).merge(target_image)
            sql_model.get_session(img_name).commit()

            print "%i: %s %s %s" % (count, searching_patch, best_state, value)
            count += 1
