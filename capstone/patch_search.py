import numpy as np

"""Input: a segment of contiguous pixels to use as a search term, plus a larger segment of contiguous pixels, which could be all of the pixels of an entire image, which is the environment to be searched. This algorithm uses RL to find the location in the larger segment which is MOST similar to the smaller segment."""

def similarity(segment, otherSegment):
    """
    A similarity metric that compares segments of the same shape, pixel-by-pixel.
    outputs a 0 to 1 similarity value.
    """
    if segment.shape != otherSegment.shape:
        assert "Segments must be of the same shape for similarity metric"
    segment = segment / 255.
    otherSegment = otherSegment / 255.
    error = np.subtract(segment, otherSegment)
    squVoxelError = error * error
    pixelError = np.sum(squVoxelError, axis = 1) / 3.
    totalError = np.sum(pixelError) / len(pixelError)
    return totalError

class RlSearch:
    """
    Searches the environment by translating and rotating the agent, checking the similarity
    between agent and the environment at the new position, and delivering a reward to the RL
    learning agent.

    Seeks to train a policy such that the search will be done more quickly with training.
    """

    self.state = np.zeros((3, 1))
    """
    x, y, and rotation of the agent.
    All zeros indicate that the agent centered in the environment and not rotated.
    """

    self.actionSet = np.zeros((5, 3)) # for a discrete search algorithm
    """
    rows: 1-step translation. (up, down, left, right, stay)
    columns: 1-step rotation. (cw, ccw, stay)  
    """
    translationStep = 1
    rotationStep = np.pi / 18.
    
    def __init__(self, agent, environment):
        """
        takes a small segment to use as a search agent, and a larger 
        segment to use as a search environment.

        Initilizes matrix V, the value per state.
        """
        self.agent = agent
        self.env = environment

        translation_state_space = (self.env.shape[1] * self.env.shape[0]) / translationStep
        self.V = np.zeros((translation_state_space, 1))

    def reward(self):
        """
        Calculates reward based on similarity between image and its environment.
        """
        test_pixels = self.get_underlying_pixels()
        sim = similarity(self.agent, test_pixels) #TODO scale this? 
        return sim        

    def translate(self, x, y):
        """        
        Moves the agent in the environment.
        """
        pass

    def rotate(self, radians):
        """
        Rotates the agent in the environment.
        """
        pass

    def get_underlying_pixels(self, state):
        """
        Given the translation and rotation of the agent, return the pixels of the environment
        that correspond to each pixel of the agent.
        """
        
        pass

    def step(self):
        """
        Selects an action, calculates next state & reward, updates V or Q.
        """
        pass

    
