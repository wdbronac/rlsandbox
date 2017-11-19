from neural_modules.ExperienceReplay.simple_exp_replay import SimpleExperienceReplay
from neural_modules.mlp import SimpleMLP
import torch
import numpy as np  # todo: replace all numpy by torch to use gpu ?
from neural_modules.ExperienceReplay.simple_exp_replay import Transition
from torch.autograd import Variable
from torch import FloatTensor
from torch import optim
from torch import ByteTensor

class SimpleDQNAgent():

    def __init__(self, env, size_exp_replay=1000, max_explo=1, min_explo=0.005, stop_explo=1000):
        self.action_size = env.action_space.shape[0]
        self.state_size = env.observation_space.shape[0]
        self.value_network = SimpleMLP(self.state_size, self.action_size)
        self.experience_replay = SimpleExperienceReplay(size_exp_replay)
        self.iteration = 0
        self.update_iteration = 0
        self.max_explo = max_explo
        self.min_explo = min_explo
        self.stop_explo = stop_explo
        self.batch_size = 64
        self.optimizer = optim.RMSprop(self.value_network.parameters())
        self.gamma = 0.99
        self.normalize_state = lambda x: (x - env.observation_space.low)/\
                                         (env.observation_space.high - env.observation_space.low)

    def act(self, state):
        return self.sample_action(state)

    def update(self, state, action, next_state, reward):
        self.experience_replay.push(self.normalize_state(state), action, self.normalize_state(next_state), reward)
        if self.update_iteration < self.batch_size:
            pass
        else:
            batch = Transition(*zip(*self.experience_replay.sample(self.batch_size)))
            # Compute a mask of non-final states and concatenate the batch elements
            non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)))

            # We don't want to backprop through the expected action values and volatile
            # will save us on temporarily changing the model parameters'
            # requires_grad to False!
            non_final_next_states = torch.stack([s for s in batch.next_state
                                                        if s is not None])
            non_final_next_states.volatile = True
            # todo: attention ca concatene les colonnes
            state_batch = torch.stack(batch.state)
            action_batch = torch.stack(batch.action)
            reward_batch = torch.stack(batch.reward)

            # Compute V(s_{t+1}) for all next states.
            next_state_values = Variable(torch.zeros(self.batch_size).type(FloatTensor))
            next_state_values[non_final_mask] = self.value_network(non_final_next_states).max(1)[0]
            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            next_state_values.volatile = False
            # Compute the expected Q values
            targets = (next_state_values * self.gamma) + reward_batch

            # todo : backrop only on action selected
            predicted = self.value_network(state_batch).gather(1, action_batch.long())
            loss = torch.sum((predicted - targets)**2)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.value_network.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
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
            _, max_action = self.value_network(Variable(FloatTensor(self.normalize_state(state)))).max(0)
            return int(max_action)

