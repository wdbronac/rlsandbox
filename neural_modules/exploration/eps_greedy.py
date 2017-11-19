import numpy as np

class EpsilonGreedy():

    def __init__(self, action_space, thumb):
        self.iteration = 0
        self.action_space = action_space
        self.thumb = thumb

    def sample(self):
        if self.iteration < self.thumb[0]:
            threshold = 1 + (self.thumb[1] - 1)/self.thumb[0] * self.iteration
        else:
            threshold = self.thumb[1]
        sample = np.random.uniform(0, 1)
        if sample < threshold:
            return 

        else:
            return random_action  # need to know action space