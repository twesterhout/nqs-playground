import torch

Net = lambda n: torch.nn.Sequential(
    torch.nn.Linear(n, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2)
)
