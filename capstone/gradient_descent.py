import image_segment
from gd_env import Env, GD_Searcher
import numpy as np
import os.path
import time

if __name__ == "__main__":
    img_name = 'butterfly'
    K = 150
    seg_data = image_segment.get_segmented_img(img_name, K)
    labels = seg_data["labels"]
    pixels = seg_data["pixels"]

    label = 10
    patch_indices = np.where(labels == label)
    patch_pixels = pixels[patch_indices]    
    indices_at_origin = (patch_indices[0] - np.min(patch_indices[0]),
                         patch_indices[1] - np.min(patch_indices[1]))

    agent = GD_Searcher(patch_pixels, indices_at_origin)
    env = Env(pixels, agent)

    print "Starting"
    state = env.init_episode()
    gamma = 100
    error = env.calculate_error(state)

    y_axis = []
    x_axis = []

    count = 0
    start = time.time()

    for i in range(400):
        gradients = env.get_gradients(state)
        scaled = gamma * gradients
        state = state - scaled
        state = env.check_state(state)            
        error_prime = env.calculate_error(state)
        improvement = error - error_prime
        error = error_prime

        y_axis.append(error)
        x_axis.append(count)
        count += 1
        #print state, error, improvement
    print time.time() - start
    print "Done!"
    import matplotlib.pyplot as plt
    plt.plot(x_axis, y_axis)
    plt.show()
