# Copyright Tom Westerhout (c) 2019
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

from numba import jit, float32, float64, complex64, complex128, void
import numpy as np
import torch

__all__ = ["logcosh"]


@jit(
    [
        void(complex64[:], complex64[:], float32[:]),
        void(complex128[:], complex128[:], float64[:]),
    ],
    nopython=True,
    fastmath=True,
)
def _log_cosh_forward_impl(z, out, tanh_x):
    """
    Kernel for implementing the forward pass of :py:class`_LogCosh`.

    :param z:   A complex 1D array, input to ``forward``.
    :param out: A complex 1D array of the same shape as ``z``. It acts the
                output buffer. Upon return from the function it will contain
                ``log(cosh(z))``.
    :param tanh_x: Precomputed ``tanh(Re[z])``.
    """
    log_2 = 0.693147180559945309417232121458176568075500134360255
    for i in range(z.size):
        x = np.abs(z[i].real)
        y = z[i].imag
        # To avoid overflow in cosh
        if x > 8:
            out[i] = x - log_2 + 0j
        else:
            out[i] = np.log(np.cosh(x)) + 0j
        out[i] += np.log(np.cos(y) + 1j * tanh_x[i] * np.sin(y))


@jit(
    [
        void(complex64[:], complex64[:], float32[:], float32[:]),
        void(complex128[:], complex128[:], float64[:], float64[:]),
    ],
    nopython=True,
    fastmath=True,
)
def _log_cosh_backward_impl(dz, out, tanh_x, tan_y):
    """
    Kernel for implementing the backward pass of :py:class`_LogCosh`.

    This function is pure magic and it's a wonder that it works.

    :param dz:  A complex 1D array, input to ``backward``.
    :param out:
        A complex 1D array of the same shape as ``z``. It acts the
        output buffer. Upon return from the function
        ``outₙ = ∂log(cosh(zₙ))/∂Re[zₙ] * Re[dzₙ] + ∂log(cosh(zₙ))/∂Im[zₙ] * Im[dzₙ]``.
    :param tanh_x: Precomputed ``tanh(Re[z])``.
    :param tan_y:  Precomputed ``tan(Im[z])``.
    """
    for i in range(len(out)):
        # B/A:
        B_over_A = tan_y[i] * tanh_x[i]
        # (∂A/∂x)/A:
        Dx_A_over_A = tanh_x[i]
        # (∂A/∂y)/A:
        Dy_A_over_A = -tan_y[i]
        # for convenience C := (∂A/∂y)/A + (B/A) * (∂A/∂x)/A:
        C = Dy_A_over_A + B_over_A * Dx_A_over_A
        # for convenience D := (∂A/∂x)/A - (B/A) * (∂A/∂y)/A:
        D = Dx_A_over_A - B_over_A * Dy_A_over_A
        # NOTE: The magic part. If you have an hour or two to spare, try checking this :)
        dx = dz[i].real
        dy = dz[i].imag
        out[i] = ((D * dx - C * dy) + 1j * (C * dx + D * dy)) / (1 + B_over_A ** 2)


class _LogCosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        """
        :return: ``log(cosh(z))``
        """
        if z.dtype == torch.float32:
            complex_type = np.complex64
        elif z.dtype == torch.float64:
            complex_type = np.complex128
        else:
            raise TypeError(
                "Supported float types are float and double, but got {}.".format(
                    z.dtype
                )
            )

        # To make sure we're not computing derivatives.
        z = z.detach()
        # x := Re[z]
        x = z.view(-1, 2)[:, 0]
        # y := Im[z]
        y = z.view(-1, 2)[:, 1]
        # Precomputing tanh(Re[z]) -- we'll need it for both forward and
        # backward passes.
        tanh_x = torch.tanh(x)
        out = torch.empty(z.size(), dtype=z.dtype, requires_grad=False)
        _log_cosh_forward_impl(
            z.view(-1).numpy().view(dtype=complex_type),
            out.view(-1).numpy().view(dtype=complex_type),
            tanh_x.numpy(),
        )
        ctx.save_for_backward(z, tanh_x)
        return out


"""
LogCosh activation function.
"""
logcosh = _LogCosh.apply
