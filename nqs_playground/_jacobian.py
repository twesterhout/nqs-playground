# Copyright Tom Westerhout (c) 2020-2021
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__all__ = ["jacobian"]


from typing import List, Optional

import torch
from torch import Tensor

if torch.has_cuda:
    import threading
    from torch.nn.parallel.scatter_gather import scatter, gather
    from torch.nn.parallel.replicate import replicate
    from torch.cuda._utils import _get_device_index
    from torch._utils import ExceptionWrapper


def jacobian(module: torch.nn.Module, parameters: List[Tensor], inputs: Tensor) -> Tensor:
    r"""Given a ``torch.nn.Module`` and a ``torch.Tensor`` of inputs, computes
    the Jacobian ∂module(inputs)/∂W where W are module's parameters.

    It is assumed that if ``inputs`` has shape ``(batch_size, in_features)``,
    then ``module(inputs)`` has shape ``(batch_size, 1)``.
    """
    return jacobian_simple(module, parameters, inputs)
    # if inputs.device.type == "cuda":
    #     return jacobian_cuda(module, inputs)
    # elif inputs.device.type == "cpu":
    #     return jacobian_cpu(module, inputs)
    # else:
    #     raise ValueError(
    #         "'inputs' tensor resides on an unsupported device: {}; expected either "
    #         "'cpu' or 'cuda'".format(inputs.device.type)
    #     )


def jacobian_simple(module: torch.nn.Module, parameters: List[Tensor], inputs: Tensor) -> Tensor:
    r"""Trivial implementation of ``jacobian``. It is used to assess
    correctness of fancier techniques.
    """
    out = inputs.new_empty(
        [inputs.size(0), sum(map(torch.numel, parameters))], dtype=parameters[0].dtype
    )
    for i in range(inputs.size(0)):
        dws = torch.autograd.grad([module(inputs[[i]])], parameters)
        torch.cat([dw.flatten() for dw in dws], out=out[i])
    return out


# def jacobian_cpu(module: torch.jit.ScriptModule, inputs: Tensor, num_threads: int = -1) -> Tensor:
#     r"""Jacobian computation on CPU."""
#     return _jacobian(module._c, inputs, num_threads=num_threads)


# def jacobian_cuda(
#     module: torch.jit.ScriptModule,
#     inputs: Tensor,
#     devices: Optional[List[torch.device]] = None,
#     output_device: Optional[torch.device] = None,
#     parallel: Optional[bool] = True,
# ) -> Tensor:
#     r"""Jacobian computation on (multiple) GPUs."""
#     if not parallel:
#         return _jacobian(module._c, inputs)
#     if devices is None:
#         device_ids = list(range(torch.cuda.device_count()))
#     else:
#         device_ids = list(map(lambda x: _get_device_index(x, True), devices))
#
#     if output_device is None:
#         output_device = inputs.device
#     inputs = scatter(inputs, device_ids, dim=0)
#     replicas = replicate(module, device_ids, detach=False)
#     outputs = _parallel_apply_jacobian(replicas, inputs)
#     return gather(outputs, output_device, dim=0)


# def _parallel_apply_jacobian(
#     replicas: List[torch.jit.ScriptModule], inputs: List[Tensor]
# ) -> List[Tensor]:
#     results = [None] * len(inputs)
#
#     def _worker(i, module, x):
#         try:
#             results[i] = _jacobian(module._c, x)
#         except Exception:
#             results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, x.device))
#
#     threads = [
#         threading.Thread(target=_worker, args=(i, m, x))
#         for i, (m, x) in enumerate(zip(replicas, inputs))
#     ]
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()
#
#     for result in results:
#         if isinstance(result, ExceptionWrapper):
#             result.reraise()
#     return results
