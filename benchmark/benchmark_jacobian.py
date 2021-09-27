import timeit
import numpy as np
import torch
from torch import Tensor

from nqs_playground import *

BATCH_SIZE = 1000
REPEAT = 5
NUMBER = 5

def make_network():
    # return torch.jit.script(torch.nn.Sequential(
    #     torch.nn.Linear(5, 6),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(6, 1, bias=False),
    # ))
    return torch.jit.script(torch.nn.Sequential(
        Unpack(30),
        torch.nn.Linear(30, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1, bias=False),
    ))


def run_gpu():
    module = make_network()
    module.to(device='cuda:0')
    inputs = torch.rand((BATCH_SIZE, 30), device='cuda:0')
    r = np.array(timeit.repeat(lambda: nqs_playground._jacobian.jacobian_cuda(module, inputs, devices=['cuda:0', 'cuda:1']), repeat=REPEAT, number=NUMBER))
    # r = np.array(timeit.repeat(lambda: jacobian(module, inputs), repeat=REPEAT, number=NUMBER))
    r *= 1000 / (BATCH_SIZE * NUMBER)
    return r


def _jacobian(module: torch.nn.Module, inputs: Tensor) -> Tensor:
    r"""Trivial implementation of ``jacobian``. It is used to assess
    correctness of fancier techniques.
    """
    parameters = list(module.parameters())
    out = inputs.new_empty(
        [inputs.size(0), sum(map(torch.numel, parameters))], dtype=parameters[0].dtype
    )
    outputs = module(inputs)
    for i in range(inputs.size(0)):
        dws = torch.autograd.grad([outputs[i]], parameters, retain_graph=True)
        torch.cat([dw.flatten() for dw in dws], out=out[i])
    return out


def run_cpu():
    module = make_network()
    inputs = torch.randint(0, 1<<30 - 1, size=(BATCH_SIZE, 8), dtype=torch.int64, device='cpu')

    r1 = _jacobian(module, inputs)
    r2 = jacobian_simple(module, inputs)
    assert torch.isclose(r1, r2, rtol=1e-4, atol=1e-6).all()

    def f():
        # return _jacobian(module, inputs)
        return jacobian_simple(module, inputs)

    r = np.array(timeit.repeat(f, repeat=REPEAT, number=NUMBER))
    r *= 1000 / (BATCH_SIZE * NUMBER)
    return r


# m = make_network()
# xs = torch.rand((BATCH_SIZE, 5), device='cpu')
# j1 = nqs_playground._jacobian.jacobian_simple(m, xs)
# 
# m.cuda()
# xs = xs.cuda()
# j2 = nqs_playground._jacobian.jacobian_cuda(m, xs, devices=['cuda:0', 'cuda:1'])
# j2 = j2.cpu()
# 
# for i in range(j1.size(0)):
#     if not torch.allclose(j1[i], j2[i]):
#         print(i)
# print(torch.allclose(j1, j2))

# print(run_gpu())
print(run_cpu())
