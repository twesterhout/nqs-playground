import torch

class Net(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        assert n == 10
        self._conv1 = torch.nn.Conv1d(
            1, 6, 10, stride=1, padding=0, dilation=1, groups=1, bias=True
        )
        self._conv2 = torch.nn.Conv1d(
            6, 6, 8, stride=1, padding=0, dilation=1, groups=1, bias=True
        )
        self._conv3 = torch.nn.Conv1d(
            6, 6, 6, stride=1, padding=0, dilation=1, groups=1, bias=True
        )
        self._dense6 = torch.nn.Linear(6, 1, bias=False)

    def forward(self, x):
        x = x.view([x.shape[0], 1, 10])

        x = torch.cat([x, x[:, :, :9]], dim=2)
        x = self._conv1(x)
        x = torch.nn.functional.relu(x)

        x = torch.cat([x, x[:, :, :7]], dim=2)
        x = self._conv2(x)
        x = torch.nn.functional.relu(x)

        x = torch.cat([x, x[:, :, :5]], dim=2)
        x = self._conv3(x)
        x = torch.nn.functional.relu(x)

        x = x.view([x.shape[0], 6, -1])
        x = x.mean(dim=2)

        x = self._dense6(x)
        x = torch.exp(x)
        return x

