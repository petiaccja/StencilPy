from frontendcf.api import index
from frontendcf.stencil import stencil
import numpy as np
import matplotlib.pyplot as plt


@stencil(jit=True)
def stencil_fill(value):
    return value


@stencil(jit=True)
def stencil_mandelbrot(c_field_real, c_field_imag, num_iters):
    c_real = c_field_real[index()]
    c_imag = c_field_imag[index()]
    zn_real = 0.0
    zn_imag = 0.0
    diverged_after = 0
    for i in range(1, num_iters):
        znsq_real = zn_real * zn_real - zn_imag * zn_imag
        znsq_imag = 2 * zn_real * zn_imag
        zn1_real = znsq_real + c_real
        zn1_imag = znsq_imag + c_imag
        mag = zn1_real * zn_real + zn1_imag * zn1_imag
        if mag < 10000.0:
            diverged_after = diverged_after + 1
        zn_real = zn1_real
        zn_imag = zn1_imag
    return diverged_after


def test_fill():
    out = np.zeros(shape=(10, 10), dtype=float)
    stencil_fill(1.0)(out)
    assert np.allclose(out, 1.0)


def test_mandelbrot():
    width = 200
    height = 100
    w_range = 4.0
    h_range = 2.0
    axis_real = (np.arange(start=0, stop=width + 1, dtype=float) / width - 0.5) * w_range
    axis_imag = (np.arange(start=0, stop=height + 1, dtype=float) / height - 0.5) * h_range
    field_real = np.reshape(axis_real, newshape=(width + 1, 1)) * np.ones(shape=(1, height + 1))
    field_imag = np.reshape(axis_imag, newshape=(1, height + 1)) * np.ones(shape=(width + 1, 1))
    out = np.zeros(shape=(width + 1, height + 1), dtype=float)
    stencil_mandelbrot(field_real, field_imag, 128)(out)
    #plt.imshow(np.transpose(out))
    #plt.show()


#test_mandelbrot()
