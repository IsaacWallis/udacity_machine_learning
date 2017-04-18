import image_segment
import numpy as np
from scipy import ndimage, misc


"""
Provides a class using RL techniques to search a large segment of pixels (such as an image)
for the patch of pixels which _best_ matches a smaller segment of pixels (such as a patch from 
a segmenting algorithm).
"""


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
        #self.state = np.array([ 0, 0 ]) 
        
        self.action = 0
        self.Q = np.random.random((self.env.shape[0], self.env.shape[1], self.action_space))
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
        sim = self.similarity(self.agent[self.indices], test_pixels) #TODO scale this? 
        return sim

    def similarity(self, patchA, patchB):
        """
        A similarity metric that compares segments of the same shape, pixel-by-pixel.
        
        :param patchA: A set of pixels
        :param patchB:
        :returns: 0 to 1 float, where 1 means the patches are identical
        :raises AssertionError: if the patch shapes are not the same
        """
        assert patchA.shape == patchB.shape
        segment = patchA / 255.
        otherSegment = patchB / 255.
        error = np.subtract(segment, otherSegment)
        squVoxelError = error * error
        pixelError = np.sum(squVoxelError, axis = 1) / 3.
        totalError = np.sum(pixelError) / len(pixelError)
        return 1.0 - totalError

    def get_underlying_pixels(self, state):
        """
        Figures out which pixels in the environment correspond to the current position
        of the pixels in the agent, to be used in similarity metrics.

        :param state: current state of the environment        
        """
        x_ndcs, y_ndcs = np.where(self.indices)
        trans_x = x_ndcs + state[0]
        trans_y = y_ndcs + state[1]
        return self.env[trans_x, trans_y]

    def choose_action(self, state):
        """
        Uses Q matrix and e-greedy to choose next action.

        :returns: an action from the available set
        """
        #TODO: make sure searcher can't go off edges of environment
        import random
        maxQ = np.max(self.Q[self.state[0], self.state[1], :])
        choices, = np.where(self.Q[self.state[0], self.state[1]] == maxQ)
        action = random.choice(choices)
        if np.random < self.epsilon: # e-greedy
            return random.choice(self.Q[state[0], state[1], :])
        else:
            return action

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
        try:
            return self.state + action_to_state[action]
        except KeyError, e:
            print "KeyError with action: ", action
        
    def search_exhaustively(self):
        import time
        from matplotlib import pyplot as plt
        heatmap = np.zeros((self.env.shape[0] - self.agent.shape[0], self.env.shape[1] - self.agent.shape[1]))
        t0 = time.time()
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                test_pixels = self.get_underlying_pixels([i,j])
                sim = self.similarity(self.agent[self.indices], test_pixels) #TODO scale this? 
                heatmap[i,j] = sim
        print("Elapsed time: ", time.time() - t0)
        fig1 = plt.figure(figsize=(5, 5))
        ax1 = fig1.add_subplot(211)
        ax1.imshow(heatmap, alpha = 1.0)

        max_ndx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        ax1.scatter(max_ndx[1], max_ndx[0], c='red', marker="x")
        print max_ndx
        
        ax2 = fig1.add_subplot(212)
        ax2.imshow(self.env, alpha = .5)

        x,y = np.where(self.indices)
        trans_x = x + self.state[0]
        trans_y = y + self.state[1]

        ax2.scatter(trans_y,
                    trans_x,
                    c = self.agent[self.indices] / 255.,
                    alpha = 1.0,
                    marker = ".",
                    s = 5)
        plt.show()

        
    def step(self):
        """
        Selects an action, calculates next state & reward, updates Q.
        """
        s = self.state
        a = self.action

        s_prime = self.get_next_state(a)
        r = self.reward(s_prime)
        print "reward ", r
        a_prime = self.choose_action(s_prime)

        a_star = np.argmax(self.Q[s_prime[0], s_prime[1]])
        if a_prime == self.Q[s_prime[0], s_prime[1], a_star]:
            a_star = a_prime            

        discount = 0.9
        learning = 0.5
        prime = 0.9
        delta = r + (discount * self.Q[s_prime[0], s_prime[1], a_star]) - self.Q[s[0], s[1], a]
        self.e[s[0], s[1], a] = self.e[s[0], s[1], a] + 1
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                for k in range(self.Q.shape[2]):
                    self.Q[i,j,k] = self.Q[s[0], s[1], k] + learning * delta * self.e[s[0], s[1], a]
                    if a_prime == a_star:
                        self.e[s[0], s[1], a] = discount * prime * self.e[s[0], s[1], a]
                    else:
                        self.e[s[0], s[1], a] = 0
        self.state = s_prime
        self.action = a_prime        
        
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

def search_smartly():
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
    searcher.run(N_Terminator(200))
    print searcher.Q
    print searcher.state
    searcher.display()
    
def search_exhaustively():
    img = ndimage.imread('butterfly.jpg')
    img = misc.imresize(img, size = 0.125)
    search_env = img
    patch_src = img
    K = 50
    labels = image_segment.segment(img, K)
    patch_indices = labels == 49

    searcher = RlSearch(search_env, patch_src, patch_indices)
    
    searcher.search_exhaustively()    

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
    search_exhaustively()
