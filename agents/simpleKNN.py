from neural_modules.ExperienceReplay.simple_exp_replay import SimpleExperienceReplay
from neural_modules.knn import SimpleKNNEvaluator
from neural_modules.mlp import SimpleMLP
import torch
import numpy as np  # todo: replace all numpy by torch to use gpu ?
from neural_modules.ExperienceReplay.simple_exp_replay import Transition
from torch.autograd import Variable
from torch import FloatTensor
from torch import optim
from torch import ByteTensor

class SimpleKNNAgent():

    def __init__(self, env, size_exp_replay=1000, max_explo=1, min_explo=0.005, stop_explo=1000):
        self.action_size = env.action_space.shape[0]
        self.state_size = env.observation_space.shape[0]
        self.knn = SimpleKNNEvaluator(self.state_size, self.action_size)
        self.iteration = 0
        self.update_iteration = 0
        self.max_explo = max_explo
        self.min_explo = min_explo
        self.stop_explo = stop_explo
        self.batch_size = 64
        self.gamma = 0.99
        self.normalize_state = lambda x: (x - env.observation_space.low)/\
                                         (env.observation_space.high - env.observation_space.low)

    # ne pas avoir d exp replay mais en fait l exp replay est integre dans la memoire du knn

    def act(self, state):
        return self.sample_action(state)

    def update(self, state, action, next_state, reward):
        self.knn.push(self.normalize_state(state), action, self.normalize_state(next_state), reward)
        self.knn.update(self.gamma)
        self.update_iteration += 1

    def sample_action(self, state):
        random_sample = np.random.rand()
        if self.iteration < self.stop_explo:
            threshold = self.max_explo - (self.max_explo - self.min_explo)/self.stop_explo * self.iteration
        else:
            threshold = self.min_explo
        self.iteration += 1
        if threshold > random_sample:
            return np.random.randint(self.action_size)
        else:
            _, max_action = self.value_network.predict(Variable(FloatTensor(self.normalize_state(state)))).max(0)
            return int(max_action)

