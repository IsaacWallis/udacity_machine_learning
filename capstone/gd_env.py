import numpy as np

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

