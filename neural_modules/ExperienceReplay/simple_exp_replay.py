from collections import namedtuple
import random
from torch.autograd import Variable
from torch import FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class SimpleExperienceReplay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(
            Variable(FloatTensor(state)),
            Variable(FloatTensor([action])),
            Variable(FloatTensor(next_state)),
            Variable(FloatTensor([reward])))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=64):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)