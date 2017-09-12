import os
import pytest

from importlib import import_module

from .util import skip_if_no_tf
from dtcwt.utils import unpack
import dtcwt
import dtcwt.compat

PRECISION_DECIMAL = 5


@skip_if_no_tf
def setup():
    global tf
    tf = import_module('tensorflow')
    dtcwt.push_backend('tf')

    # Make sure we run tests on cpu rather than gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


@skip_if_no_tf
@pytest.mark.parametrize("nlevels, include_scale", [
    (2,False),
    (2,True),
    (4,False),
    (3,True)
])
def test_scales(nlevels, include_scale):
    in_ = tf.placeholder(tf.float32, [512, 512])
    t = dtcwt.Transform2d()

    p = t.forward(in_, nlevels, include_scale)

    # At level 1, the lowpass output will be the same size as the input. At
    # levels above that, it will be half the size per level
    extent = 512 * 2**(-(nlevels-1))
    assert p.lowpass_op.get_shape().as_list() == [extent, extent]
    assert p.lowpass_op.dtype == tf.float32

    for i in range(nlevels):
        extent = 512 * 2**(-(i+1))
        assert (p.highpasses_ops[i].get_shape().as_list() ==
                [extent, extent, 6])
        assert (p.highpasses_ops[i].dtype ==
                tf.complex64)
        if include_scale:
            assert (p.scales_ops[i].get_shape().as_list() ==
                    [2*extent, 2*extent])
            assert p.scales_ops[i].dtype == tf.float32


@skip_if_no_tf
@pytest.mark.parametrize("nlevels, include_scale", [
    (2,False),
    (2,True),
    (4,False),
    (3,True)
])
def test_2d_input_tuple(nlevels, include_scale):
    in_ = tf.placeholder(tf.float32, [512, 512])
    t = dtcwt.Transform2d()
    if include_scale:
        Yl, Yh, Yscale = unpack(t.forward(in_, nlevels, include_scale), 'tf')
    else:
        Yl, Yh = unpack(t.forward(in_, nlevels, include_scale), 'tf')

    # At level 1, the lowpass output will be the same size as the input. At
    # levels above that, it will be half the size per level
    extent = 512 * 2**(-(nlevels-1))
    assert Yl.get_shape().as_list() == [extent, extent]
    assert Yl.dtype == tf.float32

    for i in range(nlevels):
        extent = 512 * 2**(-(i+1))
        assert Yh[i].get_shape().as_list() == [extent, extent, 6]
        assert Yh[i].dtype == tf.complex64
        if include_scale:
            assert Yscale[i].get_shape().as_list() == [2*extent, 2*extent]
            assert Yscale[i].dtype == tf.float32


@skip_if_no_tf
@pytest.mark.parametrize("nlevels, include_scale, batch_size", [
    (2,False,None),
    (2,True,10),
    (4,False,None),
    (3,True,2)
])
def test_batch_input(nlevels, include_scale, batch_size):
    in_ = tf.placeholder(tf.float32, [batch_size, 512, 512])
    t = dtcwt.Transform2d()
    p = t.forward_channels(in_, "nhw", nlevels, include_scale)

    # At level 1, the lowpass output will be the same size as the input. At
    # levels above that, it will be half the size per level
    extent = 512 * 2**(-(nlevels-1))
    assert p.lowpass_op.get_shape().as_list() == [batch_size, extent, extent]
    assert p.lowpass_op.dtype == tf.float32

    for i in range(nlevels):
        extent = 512 * 2**(-(i+1))
        assert (p.highpasses_ops[i].get_shape().as_list() ==
                [batch_size, extent, extent, 6])
        assert p.highpasses_ops[i].dtype == tf.complex64
        if include_scale:
            assert (p.scales_ops[i].get_shape().as_list() ==
                    [batch_size, 2*extent, 2*extent])
            assert p.scales_ops[i].dtype == tf.float32


@skip_if_no_tf
@pytest.mark.parametrize("nlevels, include_scale, batch_size", [
    (2,False,None),
    (2,True,10),
    (4,False,None),
    (3,True,2)
])
def test_batch_input_tuple(nlevels, include_scale, batch_size):
    in_ = tf.placeholder(tf.float32, [batch_size, 512, 512])
    t = dtcwt.Transform2d()

    if include_scale:
        Yl, Yh, Yscale = unpack(
            t.forward_channels(in_, "nhw", nlevels, include_scale), "tf")
    else:
        Yl, Yh = unpack(
            t.forward_channels(in_, "nhw", nlevels, include_scale), "tf")

    # At level 1, the lowpass output will be the same size as the input. At
    # levels above that, it will be half the size per level
    extent = 512 * 2**(-(nlevels-1))
    assert Yl.get_shape().as_list() == [batch_size, extent, extent]
    assert Yl.dtype == tf.float32

    for i in range(nlevels):
        extent = 512 * 2**(-(i+1))
        assert Yh[i].get_shape().as_list() == [batch_size, extent, extent, 6]
        assert Yh[i].dtype == tf.complex64
        if include_scale:
            assert (Yscale[i].get_shape().as_list() ==
                    [batch_size, 2*extent, 2*extent])
            assert Yscale[i].dtype == tf.float32


@skip_if_no_tf
@pytest.mark.parametrize("nlevels, channels", [
    (2,5),
    (2,2),
    (4,10),
    (3,6)
])
def test_multichannel(nlevels, channels):
    in_ = tf.placeholder(tf.float32, [None, 512, 512, channels])
    t = dtcwt.Transform2d()
    Yl, Yh, Yscale = unpack(
        t.forward_channels(in_, "nhwc", nlevels, include_scale=True), "tf")
    # At level 1, the lowpass output will be the same size as the input. At
    # levels above that, it will be half the size per level
    extent = 512 * 2**(-(nlevels-1))
    assert Yl.get_shape().as_list() == [None, extent, extent, channels]
    assert Yl.dtype == tf.float32

    for i in range(nlevels):
        extent = 512 * 2**(-(i+1))
        assert (Yh[i].get_shape().as_list() ==
                [None, extent, extent, channels, 6])
        assert Yh[i].dtype == tf.complex64
        assert Yscale[i].get_shape().as_list() == [
            None, 2*extent, 2*extent, channels]
        assert Yscale[i].dtype == tf.float32
