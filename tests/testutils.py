import numpy as np

from dtcwt.utils import drawedge, drawcirc, appropriate_complex_type_for

def test_complex_type_for_complex():
    assert np.issubsctype(appropriate_complex_type_for(np.zeros((2,3), np.complex64)), np.complex64)
    assert np.issubsctype(appropriate_complex_type_for(np.zeros((2,3), np.complex128)), np.complex128)

def test_complex_type_for_float():
    assert np.issubsctype(appropriate_complex_type_for(np.zeros((2,3), np.float32)), np.complex64)
    assert np.issubsctype(appropriate_complex_type_for(np.zeros((2,3), np.float64)), np.complex128)

def test_draw_circ():
    c = drawcirc(20, 4, -4, -5, 50)
    assert c.shape == (50, 50)
    assert c[45,45] == 0
    assert c[25,25] == 1

def test_draw_edge():
    e = drawedge(20, (20,20), 4, 50)
    assert e.shape == (50, 50)
