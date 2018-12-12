#!/usr/bin/env python3

from numba import jit, float32, float64, complex64, complex128, void
import numpy as np
import torch
from torch.autograd import Function


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


class _LogCosh(Function):
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
            z.numpy().view(dtype=complex_type),
            out.numpy().view(dtype=complex_type),
            tanh_x.numpy(),
        )
        ctx.save_for_backward(z, tanh_x)
        return out

        # with torch.no_grad():
        #     x_ = z.detach().numpy().view(dtype=complex_type)

        #     y  = torch.empty(z.size(), dtype=z.dtype, requires_grad=False)
        #     y_ = y.numpy().view(dtype=complex_type)

        #     _LogCosh.logcosh(x_.copy(), out=y_)
        #     ctx.save_for_backward(z.detach(), None)
        #     assert not np.any(np.isnan(y_.real)) and not np.any(np.isnan(y_.imag))
        #     assert not np.any(np.isinf(y_.real)) and not np.any(np.isinf(y_.imag))
        #     if not torch.allclose(out, y):
        #         print(out)
        #         print(y)
        #     return y

    @staticmethod
    def backward(ctx, dz):
        """
        :return: ``dlog(cosh(z)) = ∂log(cosh(z))/∂Re[z] * Re[dz] + ∂log(cosh(zₙ))/∂Im[z] * Im[dz]``.
        """
        z, tanh_x = ctx.saved_tensors
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

        x = z.view(-1, 2)[:, 0]
        y = z.view(-1, 2)[:, 1]
        tan_y = torch.tan(y)
        out = torch.empty(z.size(), dtype=z.dtype, requires_grad=False)
        _log_cosh_backward_impl(
            dz.detach().numpy().view(dtype=complex_type),
            out.numpy().view(dtype=complex_type),
            tanh_x.numpy(),
            tan_y.numpy(),
        )
        return out

        # z_ = z.detach().numpy().view(dtype=complex_type)
        # x_ = z_.real
        # y_ = z_.imag

        # dz_ = dz.detach().numpy().view(dtype=complex_type)
        # dx_ = dz_.real
        # dy_ = dz_.imag

        # tan_y_ = np.tan(y_)
        # tanh_x_ = np.tanh(x_)
        # # tanh_x_ = tanh_x.numpy()

        # # Preparation

        # # cos_y_  = np.cos(y_)
        # # sin_y_  = np.sin(y_)
        # # cosh_x_ = np.cosh(x_)
        # # sinh_x_ = np.sinh(x_)

        # # A(x_, y_) = Re[cosh(z_)] = cos(y_) * cosh(x_)
        # # B(x_, y_) = Im[cosh(z_)] = sin(y_) * sinh(x_)

        # # A_ = cos_y_ * cosh_x_
        # # B_ = sin_y_ * sinh_x_
        # B_over_A_ = tan_y_ * tanh_x_

        # # dA(x_, y_)/dx_ = cos(y_) * sinh(x_)

        # # Dx_A_ = cos_y_ * sinh_x_
        # Dx_A_over_A_ = tanh_x_

        # # dA(x_, y_)/dy_ = - sin(y_) * cosh(x_)

        # # Dy_A_ = -sin_y_ * cosh_x_
        # Dy_A_over_A_ = - tan_y_
        #
        # # C(x_, y_) = A dA/dy + B dA/dx

        # # C_ = A_ * Dy_A_ + B_ * Dx_A_
        # C_ = Dy_A_over_A_ + B_over_A_ * Dx_A_over_A_

        # # D(x_, y_) = A dA/dx - B dA/dy
        # D_ = Dx_A_over_A_ - B_over_A_ * Dy_A_over_A_

        # dF  = torch.empty(z.size(), dtype=z.dtype, requires_grad=False)
        # dF_ = dF.numpy().view(dtype=complex_type)
        #
        # dF_.real[:] = D_ * dx_ - C_ * dy_
        # dF_.imag[:] = C_ * dx_ + D_ * dy_
        # dF_.real /= (1 + B_over_A_**2)
        # dF_.imag /= (1 + B_over_A_**2)
        # assert not np.any(np.isnan(dF_.real)) and not np.any(np.isnan(dF_.imag))
        # assert not np.any(np.isinf(dF_.real)) and not np.any(np.isinf(dF_.imag))
        # return dF


"""
LogCosh activation function.
"""
logcosh = _LogCosh.apply
