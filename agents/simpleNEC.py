from neural_modules.ExperienceReplay.exp_replay import ExperienceReplay
from neural_modules.ExperienceReplay.simple_exp_replay import SimpleExperienceReplay
from neural_modules.mlp import SimpleMLP


class SimpleNECAgent():

    def __init__(self, state_size, action_size, size_exp_replay=1000, embedding_size=20, dict_length=100):
        self.embedding_size = embedding_size
        self.dict_length = dict_length
        self.exp_replay = SimpleExperienceReplay(capacity=size_exp_replay)
        self.embedding_net = SimpleMLP(state_size, embedding_size)
        self.neural_dictionary = NeuralDictionary()


    def sample_action(self, eps):



    def


    def update(self, transition):
        self.exp_replay.put(transition)
        transitions = exp_replay.sample()
        self._train(transitions)

    def _train(self, transitions):
        embedding = self.embedding_net(transitions.states)

        self.neural_dictionary.update()

