import image_segment
import numpy as np
import os.path
import time
import sys

class Env:
    """
    An environment that takes an image, distributes rewards, allows actions. Displayable.
    """
    def __init__(self, image, agent):
        self.image = image
        self.agent = agent
        
        # defines small step value for each element of the state
        self.epsilon = {
            0 : np.array([1, 0]),
            1 : np.array([0, 1])
            }
                

    def calculate_error(self, state):
        rounded =  np.rint(state).astype(np.int8)
        x_indices = self.agent.indices[0] + rounded[0]
        y_indices = self.agent.indices[1] + rounded[1]
        envPix = self.image[x_indices, y_indices, :] / 255.
        agPix = self.agent.pixels / 255.
        e = envPix - agPix
        e = e * e
        e = np.sum(e, axis = 1) / 3.
        error = np.sum(e) / len(e)
        return error

    def get_gradients(self, state):
        rounded = np.rint(state).astype(np.int8)        
        gradient_vector = np.zeros(rounded.shape)
        for index, value in enumerate(rounded):
            forward = value + self.epsilon[index]
            backward = value - self.epsilon[index]
            forward_error = self.calculate_error(forward)
            backward_error = self.calculate_error(backward)
            gradient_vector[index] = forward_error - backward_error
            #print gradient_vector
        return gradient_vector

    def init_episode(self):
        """
        :returns: a randomly selected state
        """
        agent_x, agent_y = self.agent.indices
        minX = np.min(agent_x)
        minY = np.min(agent_y)
        maxX = np.max(agent_x)
        maxY = np.max(agent_y)    
        return np.array([
            np.random.randint(minX, self.image.shape[0] - maxX),
            np.random.randint(minY, self.image.shape[1] - maxY)
        ], dtype=np.float64)

    def check_state(self, state):
        agent_x, agent_y = self.agent.indices
        minX = np.min(agent_x)
        minY = np.min(agent_y)
        maxX = np.max(agent_x)
        maxY = np.max(agent_y)
        if state[0] - minX < 0:
            state[0] = minX
        if state[0] + maxX > self.image.shape[0]:
            state[0] = maxX
        if state[1] - minY < 0:
            state[1] = minY
        if state[1] + maxY > self.image.shape[1]:
            state[1] = maxY
        return state
        
class GD_Searcher:
    def __init__(self, pixels, indices):
        self.pixels = pixels
        self.indices = indices

def gradient_descent_convergence_plot(env_pixels, patch_pixels, patch_indices):
    agent = GD_Searcher(patch_pixels, indices_at_origin)
    env = Env(env_pixels, agent)
    print "environment size:", env_pixels.shape

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

def grad_descent(env_pixels, patch_pixels, patch_indices):
    agent = GD_Searcher(patch_pixels, indices_at_origin)
    env = Env(env_pixels, agent)
    print "environment size:", env_pixels.shape

    print "Starting"
    state = env.init_episode()
    gamma = 50.
    error = env.calculate_error(state)

    count = 0
    start = time.time()

    momentum = 0.9
    velocity = np.zeros(state.shape)
    smallestError = sys.float_info.max
    bestState = state
    convergenceAtBest = 0
    for i in range(1,1000):
        gradients = env.get_gradients(state)
        velocity = (momentum * velocity) + (gamma  * gradients)
        state = state - velocity
        state = env.check_state(state)            
        error_prime = env.calculate_error(state)
        if error_prime < smallestError:
            smallestError = error_prime
            bestState = state
            convergenceAtBest = count

        improvement = error - error_prime
        error = error_prime        
        count += 1
    return bestState, smallestError, convergenceAtBest

def grad_descent_heatmap(env_pixels, patch_pixels, patch_indices):
    agent = GD_Searcher(patch_pixels, indices_at_origin)
    env = Env(env_pixels, agent)
    print "environment size:", env_pixels.shape

    print "Starting"
    state = env.init_episode()
    gamma = 50.
    error = env.calculate_error(state)

    count = 0
    start = time.time()

    momentum = 0.9
    velocity = np.zeros(state.shape)
    smallestError = sys.float_info.max
    bestState = state
    convergenceAtBest = 0
    x_ndcs = patch_indices[0]
    y_ndcs = patch_indices[1]
    heatmap = np.zeros((env_pixels.shape[0] - np.max(x_ndcs), env_pixels.shape[1] - np.max(y_ndcs)))
    for i in range(1,1000):
        gradients = env.get_gradients(state)
        velocity = (momentum * velocity) + (gamma  * gradients)
        state = state - velocity
        state = env.check_state(state)            
        error_prime = env.calculate_error(state)
        state_ndx = np.round(state).astype(np.int8)
        heatmap[state_ndx[0], state_ndx[1]] = error_prime
        if error_prime < smallestError:
            smallestError = error_prime
            bestState = state
            convergenceAtBest = count
            
        improvement = error - error_prime
        error = error_prime        
        count += 1
    print bestState, smallestError, convergenceAtBest
    return heatmap
    

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
    heatmap =  grad_descent_heatmap(pixels, patch_pixels, patch_indices)
    import patch_search
    patch_search.plot_heatmap(heatmap)

