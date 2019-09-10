import torch

Net = lambda n: torch.nn.Sequential(
    torch.nn.Linear(n, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2)
)
