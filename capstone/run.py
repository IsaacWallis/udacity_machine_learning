from scipy import ndimage, misc
import image_segment
from environment import Env
from agent import R_Learner
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
    env = Env(img)
    agent = R_Learner(patch_pixels, indices_at_origin, env)

    while True:
        state = env.get_state()
        action = agent.get_action(state)
        reward, next_state = env.take_action(action)
        agent.learn(state, action, reward, next_state)
