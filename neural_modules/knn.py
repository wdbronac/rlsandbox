import torch
from torch import norm
import torch.nn.functional as F
from torch import FloatTensor, ByteTensor

class SimpleKNNEvaluator():

    def __init__(self, state_size, action_size, n_neighbors=5, mem_size=2000):
        # idee: ne rajouter le point que si l'erreur est supérieure à un threshold
        # si c est le cas, mettre le point et supprimer un autre point: lequel ? celui ou le supprimer aurait en moyenne
        #  le moins d effet: regarder dans toute la zone qui utiliserait ce point, quelle erreur on rajouterait en l enlevant

        # peut etre ne pas faire de threshold mais regarder a chaque fois l erreur max qu il y aurait en enlevant un
        # element, et comparer à l'erreur de ne pas rajouter le nouvel element
        # pour chaque point environ, memoriser l erreur  a peu pres

        # a ameliorer: il va y avoir des trucs pas bons au debut parce qu il va y avoir des zeros dans les knn mais ca devrait s enlever apres
        self.mem_size = mem_size
        self.n_neighbors = n_neighbors
        self.action_size = action_size
        self.state_size = state_size
        self.memory_states = [0 for _ in range(action_size)]
        self.v_mem = [torch.zeros(mem_size, 1) for _ in range(action_size)]  # for every s, a, the estimated value: list containing as much tensors as actions
        self.s_mem = torch.zeros(mem_size, state_size)  # memory of states
        self.a_mem = torch.zeros(mem_size, 1)  # memory of actions
        self.r_mem = torch.zeros(mem_size, 1)  # memory of rewards
        self.ns_mem = torch.zeros(mem_size, state_size)  # memory of next states
        self.cursor = 0
        # self.finals_mem = ByteTensor(mem_size)

    def update(self, gamma):
        targets = [torch.zeros(self.mem_size, 1) for _ in range(self.action_size)]
        for a in range(self.action_size):
            targets[self.a_mem == a] = self.r_mem[self.a_mem == a] + gamma * self.predict(self.s_mem)
        self.v_mem = targets

    def _Q_app(self, s, a):
        indices = F.cosine_similarity(s[:, None], self.s_mem[self.a_mem == a][None], 2).topk(self.n_neighbors)[1]
        value = self.v_mem[a][indices, :].mean(-2)
        return value

    def predict(self, state):
        return max(self._Q_app(state, a) for a in range(self.action_size))

    def push(self, state, action, next_state, reward):
        self.s_mem[self.cursor, :] = FloatTensor(state)
        self.a_mem[self.cursor, :] = action
        self.r_mem[self.cursor, :] = reward
        self.ns_mem[self.cursor, :] = FloatTensor(next_state)
        self.cursor = (self.cursor + 1) % self.mem_size

