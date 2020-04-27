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


def run_cpu():
    module = make_network()
    inputs = torch.randint(0, 1<<30 - 1, size=(BATCH_SIZE, 8), dtype=torch.int64, device='cpu')

    def f():
        return jacobian_cpu(module, inputs, num_threads=-1)

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
