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
    
    def __init__(self, search_env, search_patch_src, patch_indices):
        """
        Initializes matrices V and Q, 

        :param agent: a patch of pixels to use as search term
        :param environment: a larger patch of pixels to search over
        """

        self.env = search_env

        bbox = self.get_bbox(patch_indices)
        self.indices = patch_indices[bbox[0] : bbox[1] , bbox[2] : bbox[3]]
        self.agent = search_patch_src[bbox[0] : bbox[1] , bbox[2] : bbox[3]]
        
        self.bbox = np.where(self.indices)
        self.action_space = 5 #up, down, left, right, stay

        #state initialized to center of image
        self.state = np.array([ self.env.shape[0] / 2, self.env.shape[1] / 2 ]) 
        
        self.action = 0
        self.Q = np.zeros((self.env.shape[0], self.env.shape[1], self.action_space))
        self.e = np.zeros((self.env.shape[0], self.env.shape[1], self.action_space))

        self.epsilon = 0.1

    def get_bbox(self, patch_indices):
        a = np.where(patch_indices)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return bbox
            
    def reward(self, state):
        """
        Calculates reward based on similarity between image and its environment.
        """
        test_pixels = self.get_underlying_pixels(state)
        #print test_pixels
        #sim = similarity(self.agent, test_pixels) #TODO scale this? 
        #return sim        

    def get_underlying_pixels(self, state):
        """
        Figures out which pixels in the environment correspond to the current position
        of the pixels in the agent, to be used in similarity metrics.

        :param state: current state of the environment        
        """
        x_trans = self.indices[0] + self.state[0]
        y_trans = self.indices[1] + self.state[1]
        return self.env[x_trans, y_trans]

    def choose_action(self, state):
        """
        Uses Q matrix and e-greedy to choose next action.

        :returns: an action from the available set
        """
        #TODO: make sure searcher can't go off edges of environment
        maxA = np.argmax(self.Q[state, :])
        if np.random < self.epsilon: # e-greedy
            return random.choice(self.Q[state[0], state[1], :])
        else:
            return maxA

    def get_next_state(self, action):
        """        
        :param action: The action that is changing the state
        :returns: the new state
        """
        action_to_state = {
            0 : (0, 0), #stay
            1 : (1, 0), #up
            2 : (-1, 0), #down
            3 : (0, 1), #right
            4 : (0, -1), #left
        }
        return self.state + action_to_state[action]
        
    def step(self):
        """
        Selects an action, calculates next state & reward, updates V or Q.
        """
        print "step"
        a = self.choose_action(self.state)
        s_prime = self.get_next_state(a)
        r_prime = self.reward(s_prime)
                
    def run(self, terminator):
        """
        Runs steps until termination.

        :terminator: A function that return True when termination has been reached.
        """
        self.step()
        if not terminator():
            self.run(terminator)

    def display(self):
        from matplotlib import pyplot as plt
        import matplotlib.patches as mpatch

        plt.figure(figsize=(5, 5))
        plt.imshow(self.env, alpha = .5)

        x,y = np.where(self.indices)
        trans_x = x + self.state[0]
        trans_y = y + self.state[1]

        plt.scatter(trans_y,
                    trans_x,
                    c = self.agent[self.indices] / 255.,
                    alpha = 1.0,
                    marker = ".",
                    s = 5)
        plt.show()


class N_Terminator:
    """
    Terminator class to use as input to RLSearch.run()
    """
    def __init__(self, n):
        """
        :param n: the number of times this terminator can be called before it returns true.
        """
        self.n = n
        self.inc = 0
    def __call__(self):
        self.inc = self.inc + 1
        return self.inc > self.n
        
if __name__ == "__main__":
    import image_segment
    import numpy as np
    from scipy import ndimage, misc

    img = ndimage.imread('butterfly.jpg')
    img = misc.imresize(img, size = 0.0625)
    
    env = ndimage.imread('box-turtle.jpeg')
    env = misc.imresize(env, size = 0.25)

    search_env = img
    patch_src = img
    K = 50
    labels = image_segment.segment(img, K)
    patch_indices = labels == 15

    searcher = RlSearch(search_env, patch_src, patch_indices)
    searcher.run(N_Terminator(10))
    searcher.display()
