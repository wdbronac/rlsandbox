from torch import FloatTensor


class SimpleNeuralDictionary():

    def __init__(self, memory_size):
        self.memory = None
        self.max_size = memory_size

    def update_dict(self, embeddings, targets): # attention les targets doivent surement rester fixes
        if self.memory is None:
            self.memory = FloatTensor(embeddings)
        pass

    def compute_value(self, embeddings):
        projections = embeddings.matmul(memory)