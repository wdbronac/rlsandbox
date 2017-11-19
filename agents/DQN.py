from neural_modules.ExperienceReplay.exp_replay import ExperienceReplay


class DQNAgent():

    def __init__(self, params):
        self.action_space = params.action_space
        self.value_network = NeuralNet(params.neural_net)
        self.experience_replay = ExperienceReplay(params.exp_replay)
        self.exploration = EpsilonGreedy(eps_greedy_parameters, action_space)

    def act(self, state):
        return self.exploration.sample()

    def update(self, transition):
        self.experience_replay.put(transition)
        transitions = self.experience_replay.sample()
        target = transition.reward + max(self.value_network(transition.next_state))
        # todo : backrop only on action selected
        loss = square(self.value_network(state)[action] - target)
        loss.backward()

