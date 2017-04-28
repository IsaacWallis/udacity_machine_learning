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

    label = 12
    patch_indices = np.where(labels == label)
    patch_pixels = pixels[patch_indices]    
    indices_at_origin = (patch_indices[0] - np.min(patch_indices[0]),
                         patch_indices[1] - np.min(patch_indices[1]))

    agent = GD_Searcher(patch_pixels, indices_at_origin)
    env = Env(pixels, agent)
    print "environment size:", pixels.shape

    print "Starting"
    state = env.init_episode()
    gamma = 50.
    error = env.calculate_error(state)

    y_axis = []
    x_axis = []

    count = 0
    start = time.time()

    momentum = 0.9
    velocity = np.zeros(state.shape)
    for i in range(1,1000):
        gradients = env.get_gradients(state)
        velocity = (momentum * velocity) + (gamma  * gradients)
        state = state - velocity
        state = env.check_state(state)            
        error_prime = env.calculate_error(state)
        improvement = error - error_prime
        error = error_prime
        y_axis.append(error)
        x_axis.append(count)
        count += 1

    print time.time() - start
    print "Done!"
    import matplotlib.pyplot as plt
    plt.plot(x_axis, y_axis)
    plt.show()
