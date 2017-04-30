import image_segment
import numpy as np
import os.path
import time
import sys

class Env:
    """
    An environment that takes an image, distributes rewards, allows actions. Displayable.
    """
    def __init__(self, image, agent_pixels, agent_indices):
        self.image = image
        self.agent_pixels = agent_pixels
        self.agent_indices = agent_indices
        self.heatmap = np.zeros((env_pixels.shape[0] - np.max(agent_indices[0]),
                                 env_pixels.shape[1] - np.max(agent_indices[1])))        
        # defines small step value for each element of the state
        self.epsilon = {
            0 : np.array([1, 0]),
            1 : np.array([0, 1])
            }
                

    def calculate_error(self, state):
        rounded =  np.rint(state).astype(np.int8)
        x_indices = self.agent_indices[0] + rounded[0]
        y_indices = self.agent_indices[1] + rounded[1]
        envPix = self.image[x_indices, y_indices, :] / 255.
        agPix = self.agent_pixels / 255.
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

    def get_random_state(self):
        """
        :returns: a randomly selected state
        """
        agent_x, agent_y = self.agent_indices
        minX = np.min(agent_x)
        minY = np.min(agent_y)
        maxX = np.max(agent_x)
        maxY = np.max(agent_y)    
        return np.array([
            np.random.randint(minX, self.image.shape[0] - maxX),
            np.random.randint(minY, self.image.shape[1] - maxY)
        ], dtype=np.float64)

    def check_state(self, state):
        agent_x, agent_y = self.agent_indices
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

    def start_agent(self, initState = None, gamma = 50., momentum = 0.9, maxT = 1000, plot = False):
        
        state = initState
        if initState is None:
            state = self.get_random_state()
            
        error = self.calculate_error(state)

        count = 0
        start = time.time()

        velocity = np.zeros(state.shape)
        smallestError = sys.float_info.max
        bestState = state
        convergenceAtBest = 0
        
        error_axis = []
        time_axis = []

        for i in range(maxT):
            gradients = self.get_gradients(state)
            velocity = (momentum * velocity) + (gamma  * gradients)
            state = state - velocity
            state = self.check_state(state)            
            error_prime = self.calculate_error(state)
            state_ndx = np.round(state).astype(np.int8)
            self.heatmap[state_ndx[0], state_ndx[1]] = error_prime
            if plot:
                error_axis.append(error_prime)
                time_axis.append(count)
            if error_prime < smallestError:
                smallestError = error_prime
                bestState = state
                convergenceAtBest = count

            improvement = error - error_prime
            error = error_prime        
            count += 1
        if plot:    
            import matplotlib.pyplot as plt
            plt.plot(time_axis, error_axis)
            plt.show()
        return bestState, smallestError, convergenceAtBest        

    def get_heatmap(self):
        return self.heatmap
    
def multi_gradient_descent(env_pixels, patch_pixels, patch_indices):
    env = Env(env_pixels, patch_pixels, patch_indices)
    initStateA = env.get_random_state()
    initStateB = env.get_random_state()
    stateA, rewardA, tA = env.start_agent(initState = initStateA, maxT = 50)
    stateB, rewardB, tB = env.start_agent(initState = initStateB, maxT = 50)
    for i in range(20):        
        if rewardA < rewardB:
            stateB, rewardB, tB = env.start_agent( maxT = 50)
        elif rewardB < rewardA:
            stateA, rewardA, tA = env.start_agent( maxT = 50)
            
    import patch_search
    patch_search.plot_heatmap(env.get_heatmap())

if __name__ == "__main__":
    img_name = 'butterfly'
    K = 150
    seg_data = image_segment.get_segmented_img(img_name, K)
    labels = seg_data["labels"]
    env_pixels = seg_data["pixels"]

    label = 12
    patch_indices = np.where(labels == label)
    patch_pixels = env_pixels[patch_indices]    
    indices_at_origin = (patch_indices[0] - np.min(patch_indices[0]),
                         patch_indices[1] - np.min(patch_indices[1]))
    print multi_gradient_descent(env_pixels, patch_pixels, indices_at_origin)
    #heatmap =  grad_descent_heatmap(env_pixels, patch_pixels, patch_indices)
    #import patch_search
    #patch_search.plot_heatmap(heatmap)

    #max_ndx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    #patch_search.display_patch(env_pixels, patch_pixels, patch_indices, max_ndx)
