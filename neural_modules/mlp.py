from torch import nn
from torch.nn import Linear
from torch.nn.functional import relu


class SimpleMLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lin1 = nn.Linear(self.input_size, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        return self.lin2(relu(self.lin1(x)))

