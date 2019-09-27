import torch

class Net(torch.jit.ScriptModule):
    def __init__(self, n: int):
        super().__init__()
        self._conv1 = torch.nn.Conv2d(
            1, 10, 4, stride=1, padding=0, dilation=1, groups=1, bias=True
        )
        self._conv2 = torch.nn.Conv2d(
            10, 10, 3, stride=1, padding=0, dilation=1, groups=1, bias=True
        )
        self._conv3 = torch.nn.Conv2d(
            10, 10, 2, stride=1, padding=0, dilation=1, groups=1, bias=True
        )
        self._dense6 = torch.nn.Linear(10, 1, bias=False)

    @torch.jit.script_method
    def forward(self, x):
        x = x.view([x.shape[0], 1, 4, 4])

        x = torch.cat([x, x[:, :, :3, :]], dim=2)
        x = torch.cat([x, x[:, :, :, :3]], dim=3)
        x = self._conv1(x)
        x = torch.nn.functional.relu(x)

        x = torch.cat([x, x[:, :, :2, :]], dim=2)
        x = torch.cat([x, x[:, :, :, :2]], dim=3)
        x = self._conv2(x)
        x = torch.nn.functional.relu(x)

        x = torch.cat([x, x[:, :, :1, :]], dim=2)
        x = torch.cat([x, x[:, :, :, :1]], dim=3)
        x = self._conv3(x)
        x = torch.nn.functional.relu(x)

        x = x.view([x.shape[0], 10, -1])
        x = x.mean(dim=2)

        x = self._dense6(x)
        # x = torch.nn.functional.softplus(x)
        x = torch.exp(x)
        return x
