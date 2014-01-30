import itertools
import numpy as np
from six.moves import xrange

from dtcwt.utils import *

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

def test_stacked_2d_matrix_vector_product():
    nr, nc = (30, 20)
    mr, mc = (3, 4)

    # Generate a random stack of 2d matrices and 2d vectors
    matrices = np.random.rand(nr, nc, mr, mc)
    vectors = np.random.rand(nr, nc, mc)

    # Compute product to test
    out = stacked_2d_matrix_vector_prod(matrices, vectors)

    # Check output shape is what we expect
    assert out.shape == (nr, nc, mr)

    # Check each product
    for r, c in itertools.product(xrange(nr), xrange(nc)):
        m = matrices[r, c, :, :]
        v = vectors[r, c, :]
        o = out[r, c, :]

        assert m.shape == (mr, mc)
        assert v.shape == (mc,)
        assert o.shape == (mr,)

        gold = m.dot(v)
        max_delta = np.abs(gold - o).max()
        assert max_delta < 1e-8

def test_stacked_2d_vector_matrix_product():
    nr, nc = (30, 20)
    mr, mc = (3, 4)

    # Generate a random stack of 2d matrices and 2d vectors
    matrices = np.random.rand(nr, nc, mr, mc)
    vectors = np.random.rand(nr, nc, mr)

    # Compute product to test
    out = stacked_2d_vector_matrix_prod(vectors, matrices)

    # Check output shape is what we expect
    assert out.shape == (nr, nc, mc)

    # Check each product
    for r, c in itertools.product(xrange(nr), xrange(nc)):
        m = matrices[r, c, :, :]
        v = vectors[r, c, :]
        o = out[r, c, :]

        assert m.shape == (mr, mc)
        assert v.shape == (mr,)
        assert o.shape == (mc,)

        gold = v.dot(m)
        max_delta = np.abs(gold - o).max()
        assert max_delta < 1e-8

def test_stacked_2d_matrix_matrix_product():
    nr, nc = (30, 20)
    mr1, k, mc2 = (3, 4, 5)

    # Generate a random stack of 2d matrices and 2d vectors
    matrices1 = np.random.rand(nr, nc, mr1, k)
    matrices2 = np.random.rand(nr, nc, k, mc2)

    # Compute product to test
    out = stacked_2d_matrix_matrix_prod(matrices1, matrices2)

    # Check output shape is what we expect
    assert out.shape == (nr, nc, mr1, mc2)

    # Check each product
    for r, c in itertools.product(xrange(nr), xrange(nc)):
        m1 = matrices1[r, c, :, :]
        m2 = matrices2[r, c, :, :]
        o = out[r, c, :, :]

        assert m1.shape == (mr1, k)
        assert m2.shape == (k, mc2)
        assert o.shape == (mr1, mc2)

        gold = m1.dot(m2)
        max_delta = np.abs(gold - o).max()
        assert max_delta < 1e-8

