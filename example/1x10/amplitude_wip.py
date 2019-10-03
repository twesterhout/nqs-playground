import torch


class Exp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


Net = lambda n: torch.nn.Sequential(
    torch.nn.Linear(n, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1, bias=False),
    Exp(),
)
