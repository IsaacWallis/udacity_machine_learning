"""
Provides a class using RL techniques to search a large segment of pixels (such as an image)
for the patch of pixels which _best_ matches a smaller segment of pixels (such as a patch from 
a segmenting algorithm).
"""
import numpy as np

def similarity(patchA, patchB):
    """
    A similarity metric that compares segments of the same shape, pixel-by-pixel.

    :param patchA: A set of pixels
    :param patchB:
    :returns: 0 to 1 float, where 1 means the patches are identical
    :raises AssertionError: if the patch shapes are not the same
    """
    assert patchA.shape == patchB.shape
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

    state = np.zeros((3, 1))
    """
    x, y, and rotation of the agent.
    All zeros indicate that the agent centered in the environment and not rotated.
    """

    actionSet = np.zeros((5, 3)) # for a discrete search algorithm
    """
    rows: 1-step translation. (up, down, left, right, stay)
    columns: 1-step rotation. (cw, ccw, stay)  
    """
    
    def __init__(self, agent, environment):
        """
        Initializes matrices V and Q, 

        :param agent: a patch of pixels to use as search term
        :param environment: a larger patch of pixels to search over
        """
        self.agent = agent
        self.env = environment

        self.translationStep = 1
        self.rotationStep = np.pi / 18.

        self.translation_state_space = (self.env.shape[1] * self.env.shape[0]) / self.translationStep
        self.V = np.zeros((self.translation_state_space, 1))

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

    
if __name__ == "__main__":
    import image_segment
    import cv2
    import numpy as np
    img = cv2.imread('butterfly.jpg')
    img = cv2.resize(img, (0,0), fx = 0.0625, fy = 0.0625)

    K = 50
    labels = image_segment.segment(img, K)
    patch_ndx = 2
    agent = image_segment.segment_as_patch(img, labels, patch_ndx)
    env = image_segment.image_as_patch(img)

    #print env.ij
    #searcher = RlSearch(agent, img)
