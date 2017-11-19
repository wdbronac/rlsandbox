from neural_modules.ExperienceReplay.exp_replay import ExperienceReplay

class NecAgent():

    def __init__(self):
        self.exp_replay = ExperienceReplay()
        self.embedding_net = CNN()
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

