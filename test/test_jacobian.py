import torch
from torch import Tensor

from nqs_playground import jacobian_simple, jacobian_cpu, jacobian_cuda



def make_dense_network():
    return torch.jit.script(torch.nn.Sequential(
        torch.nn.Linear(20, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1, bias=False),
    ))

def test_cpu():
    for i in range(7):
        m = make_dense_network()
        for j in range(7):
            x = torch.rand(57, 20)
            j_expected = jacobian_simple(m, x)
            j_predicted = jacobian_cpu(m, x)
            assert torch.allclose(j_expected, j_predicted)

def test_cuda():
    for i in range(7):
        m = make_dense_network()
        for j in range(7):
            x = torch.rand(57, 20)
            j_expected = jacobian_simple(m, x)
            m.cuda()
            x = x.cuda()
            j_predicted = jacobian_cuda(m, x).cpu()
            m.cpu()
            x = x.cpu()
            try:
                assert torch.allclose(j_expected, j_predicted)
            except AssertionError:
                torch.jit.save(m, "bad_network.pth")
                torch.save(x, "bad_input.pth")
                raise

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# 
# def debug():
#     m = torch.jit.load("bad_network.pth")
#     x = torch.load("bad_input.pth")
#     m.cuda()
#     x = x.cuda()
#     j_expected = jacobian_simple(m, x).cpu()
#     j_predicted = jacobian_cuda(m, x).cpu()
#     for i in range(j_expected.size(0)):
#         if not torch.allclose(j_expected[i], j_predicted[i]):
#             for j in range(j_expected.size(1)):
#                 if not torch.allclose(j_expected[i, j], j_predicted[i, j]):
#                     print(i, j, j_expected[i, j].tolist(), j_predicted[i, j].tolist())
# 
# 
# debug()
