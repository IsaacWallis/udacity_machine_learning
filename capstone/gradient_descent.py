import sys
import time
import numpy as np
import image_segment


class Env:
    """
    An environment that takes an image, distributes rewards, allows actions. Displayable.
    """

    def __init__(self, image, agent_pixels, agent_indices):
        self.image = image
        self.agent_pixels = agent_pixels
        self.agent_indices = agent_indices
        self.heatmap = np.zeros((image.shape[0] - np.max(agent_indices[0]),
                                 image.shape[1] - np.max(agent_indices[1])))
        # defines small step value for each element of the state
        self.epsilon = {
            0: np.array([1, 0]),
            1: np.array([0, 1])
        }

    def calculate_error(self, state):
        rounded = np.rint(state).astype(np.int8)
        x_indices = self.agent_indices[0] + rounded[0]
        y_indices = self.agent_indices[1] + rounded[1]
        envPix = self.image[x_indices, y_indices, :] / 255.
        agPix = self.agent_pixels / 255.
        e = envPix - agPix
        e = e * e
        e = np.sum(e, axis=1) / 3.
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
        return gradient_vector

    def get_random_state(self):
        """
        :returns: a randomly selected state
        """
        agent_x, agent_y = self.agent_indices
        max_x, max_y, min_x, min_y = self.get_bounding_box(agent_x, agent_y)
        return np.array([
            np.random.randint(min_x, self.image.shape[0] - max_x),
            np.random.randint(min_y, self.image.shape[1] - max_y)
        ], dtype=np.float64)

    @staticmethod
    def get_bounding_box(indices_x, indices_y):
        min_x = np.min(indices_x)
        min_y = np.min(indices_y)
        max_x = np.max(indices_x)
        max_y = np.max(indices_y)
        return max_x, max_y, min_x, min_y

    def check_state(self, state):
        agent_x, agent_y = self.agent_indices
        max_x, max_y, min_x, min_y = self.get_bounding_box(agent_x, agent_y)
        if state[0] - min_x < 0:
            state[0] = min_x
        if state[0] + max_x > self.image.shape[0]:
            state[0] = max_x
        if state[1] - min_y < 0:
            state[1] = min_y
        if state[1] + max_y > self.image.shape[1]:
            state[1] = max_y
        return state

    def start_agent(self, initState=None, gamma=50., momentum=0.9, maxT=1000, plot=False):

        state = initState
        if initState is None:
            state = self.get_random_state()

        error = self.calculate_error(state)

        count = 0
        start = time.time()

        velocity = np.zeros(state.shape)
        smallest_error = sys.float_info.max
        best_state = state
        convergence_at_best = 0

        error_axis = []
        time_axis = []

        for i in range(maxT):
            gradients = self.get_gradients(state)
            velocity = (momentum * velocity) + (gamma * gradients)
            state = state - velocity
            state = self.check_state(state)
            error_prime = self.calculate_error(state)
            state_ndx = np.round(state).astype(np.int8)
            self.heatmap[state_ndx[0], state_ndx[1]] = error_prime
            if plot:
                error_axis.append(error_prime)
                time_axis.append(count)
            if error_prime < smallest_error:
                smallest_error = error_prime
                best_state = state
                convergence_at_best = count

            improvement = error - error_prime
            error = error_prime
            count += 1
        if plot:
            plt.plot(time_axis, error_axis)
            plt.show()
        return best_state, smallest_error, convergence_at_best

    def get_heatmap(self):
        return self.heatmap


def multi_gradient_descent(env_pixels, patch_pixels, patch_indices, num_agents):
    env = Env(env_pixels, patch_pixels, patch_indices)
    agent_dict = {}
    max_t = 150
    import pandas as pd
    data_frame = pd.DataFrame(columns=["s", "s_prime", "r_prime", "t_prime"])
    for i in range(num_agents):
        agent_dict[i] = {}
        initial_state = env.get_random_state()
        state, reward, t = env.start_agent(initState=initial_state, maxT=max_t)
        data_frame.loc[i] = [initial_state, state, reward, t]
    for i in range(100):
        best = data_frame["r_prime"].idxmax()
        worst = data_frame["r_prime"].idxmin()
        best_init_state = data_frame.iloc[best]["s"]
        worst_found_state = data_frame.iloc[worst]["s_prime"]
        avg = (best_init_state + worst_found_state) / 2
        state, reward, t = env.start_agent(initState=avg, maxT=max_t)
        data_frame.iloc[worst] = [avg, state, reward, t]
    best_final_index = data_frame["r_prime"].idxmax()
    data_frame.iloc[best_final_index]

    import patch_search
    patch_search.plot_heatmap(env.heatmap)
    return data_frame.iloc[best_final_index]["s_prime"], data_frame.iloc[best_final_index]["r_prime"]


if __name__ == "__main__":
    img_name = 'butterfly'
    K = 150
    seg_data = image_segment.get_segmented_image(img_name, K)
    labels = seg_data["labels"]
    env_pixels = seg_data["pixels"]

    label = 12
    patch_indices = np.where(labels == label)
    patch_pixels = env_pixels[patch_indices]
    indices_at_origin = (patch_indices[0] - np.min(patch_indices[0]),
                         patch_indices[1] - np.min(patch_indices[1]))
    print multi_gradient_descent(env_pixels, patch_pixels, indices_at_origin, 5)
    # heatmap =  grad_descent_heatmap(env_pixels, patch_pixels, patch_indices)
    # import patch_search
    # patch_search.plot_heatmap(heatmap)

    # max_ndx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    # patch_search.display_patch(env_pixels, patch_pixels, patch_indices, max_ndx)
