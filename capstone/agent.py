import numpy as np
import pandas as pd

class R_Learner:
    def __init__(self, pixels, indices, env):
        self.pixels = pixels
        self.indices = indices
        self.env = env
        self.p = 0
        self.q = pd.DataFrame(columns = env.action_space)
        self.epsilon = 0.1
        env.add_agent(self)
        self.alpha = 0.1
        self.beta = 0.1
        
    def get_action(self, state):
        available = self.env.get_available_actions()
        self.init_state_if_necessary(state)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(available)
        else:
            row = self.q.loc[[state]]
            row_v = row.values.flatten()
            mx = np.max(row_v)
            idx = np.where(row_v == mx)
            actions = row.columns.values[idx]
            action = np.random.choice(actions)
        return action

    def init_state_if_necessary(self, state):
        if state not in self.q.index:
            row = pd.Series([0] * len(self.env.action_space), index = self.q.columns, name=state)
            self.q = self.q.append(row)            
    
    def learn(self, s, a, r, s_prime):
        self.init_state_if_necessary(s_prime)
        eq_2 = r - self.p + np.max(self.q.loc[[s_prime]].values) - self.q.get_value(s, a)
        self.q.loc[s, a] = self.q.get_value(s, a) + self.alpha * (eq_2)
        if np.max(self.q.loc[[s]].values) == self.q.get_value(s,a):
            self.p = self.p + self.beta * (eq_2)
