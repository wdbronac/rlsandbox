class ExperienceReplay():

    def __init__(self):
        self.transitions = []

    def put(self, transition):
        self.transitions.put(transition)

    def sample(self):
        return self.transitions[0]