from scipy import ndimage, misc
import image_segment
from gd_env import Env, GD_Searcher
import numpy as np

if __name__ == "__main__":
    img = ndimage.imread('butterfly.jpg')
    img = misc.imresize(img, size = 0.0625)

    search_env = img
    patch_src = img
    K = 50
    labels = image_segment.segment(img, K)
    label = 15
    patch_indices = np.where(labels == label)
    patch_pixels = img[patch_indices]
    
    indices_at_origin = (patch_indices[0] - np.min(patch_indices[0]),
                         patch_indices[1] - np.min(patch_indices[1]))

    agent = GD_Searcher(patch_pixels, indices_at_origin)
    env = Env(img, agent)


    print "Starting"
    state = env.init_episode()
    gamma = .01
    error = env.calculate_error(state)
    try:
        while True:
            gradients = env.get_gradients(state)
            scaled = gamma * gradients
            state = state - scaled
            state = env.check_state(state)            
            error_prime = env.calculate_error(state)
            improvement = error_prime - error
            error = error_prime
            print state, error, improvement
    except KeyboardInterrupt:
        print "Done!"
