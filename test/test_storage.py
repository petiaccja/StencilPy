from stencilpy.storage import Field
from stencilpy.concepts import Dimension
import numpy as np

IDim = Dimension()
JDim = Dimension()


def test_dimensions():
    assert IDim.id != JDim.id
    assert IDim != JDim


def test_field_moveaxis():
    data = np.array([[1, 2], [3, 4]])
    a = Field([IDim, JDim], data)
    b = Field([JDim, IDim], np.transpose(data))

    assert np.allclose(a.data, b.data)
    assert a.sorted_dimensions == b.sorted_dimensions


def test_field_elementwise():
    a = Field([IDim, JDim], np.array([[1, 2], [3, 4]]))
    b = Field([IDim, JDim], np.array([[1.2, 1.3], [1.4, 1.5]]))
    r = a + b
    expected = Field([IDim, JDim], np.array([[2.2, 3.3], [4.4, 5.5]]))
    assert np.allclose(r.data, expected.data)


def test_field_slicing():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inp = Field([IDim, JDim], data)
    expected = Field([IDim, JDim], data[1::, 0:2])
    result = inp[JDim[0:2], IDim[1::]]
    assert np.all(result.data == expected.data)
