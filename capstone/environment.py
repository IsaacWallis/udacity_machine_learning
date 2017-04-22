import numpy as np

class Env:
    """
    An environment that takes an image, distributes rewards, allows actions. Displayable.
    """
    def __init__(self, image):
        self.image = image
        self.state = np.array([image.shape[0] / 2, image.shape[1] / 2], dtype = np.int8)
        self.action_space = ['u', 'd', 'l', 'r']

        self.action_to_state = {
            'u' : ( 0, 1),
            'd' : ( 0,-1),
            'r' : ( 1, 0),
            'l' : (-1, 0)
        }
        
    def add_agent(self, agent):
        self.agent = agent

    def get_available_actions(self):
        """
        :returns: any actions that would not result in the agent going off the edge of the environment.
        """
        agent_x, agent_y = self.agent.indices
        minX = np.min(agent_x)
        minY = np.min(agent_y)
        maxX = np.max(agent_x)
        maxY = np.max(agent_y)
        av = []
        if self.state[0] + minX >= 0:
            av.append('l')
        if self.state[0] + maxX <= self.image.shape[0]:
            av.append('r')
        if self.state[1] + minY >= 0:
            av.append('u')
        if self.state[1] + maxY <= self.image.shape[1]:
            av.append('d')
        return av
        
    def get_state(self):
        return tuple(self.state)

    def take_action(self, action):
        """
        :returns: reward, next state
        """
        next_state = self.state + self.action_to_state[action]
        x_indices = self.agent.indices[0] + self.state[0]
        y_indices = self.agent.indices[1] + self.state[1]
        envPix = self.image[x_indices, y_indices]# / 255.
        agPix = self.agent.pixels# / 255.
        e = envPix - agPix
        e = e * e
        e = np.sum(e, axis = 1) / 3.
        error = np.sum(e) / len(e)
        return -error, tuple(next_state)


