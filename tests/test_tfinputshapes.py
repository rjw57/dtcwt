import os
import pytest

from importlib import import_module

from .util import skip_if_no_tf
PRECISION_DECIMAL = 5


@skip_if_no_tf
def test_setup():
    global tf, Transform2d, dtwavexfm2, dtwaveifm2
    tf = import_module('tensorflow')
    dtcwt_tf = import_module('dtcwt.tf')
    dtcwt_tf_xfm2 = import_module('dtcwt.tf.transform2d')
    Transform2d = getattr(dtcwt_tf, 'Transform2d')
    dtwavexfm2 = getattr(dtcwt_tf_xfm2, 'dtwavexfm2')
    dtwaveifm2 = getattr(dtcwt_tf_xfm2, 'dtwaveifm2')

    # Make sure we run tests on cpu rather than gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


@skip_if_no_tf
@pytest.mark.parametrize("nlevels, include_scale", [
    (2,False),
    (2,True),
    (4,False),
    (3,True)
])
def test_2d_input(nlevels, include_scale):
    in_ = tf.placeholder(tf.float32, [512, 512])
    t = Transform2d()
    # Calling forward with a 2d input will throw a warning
    p = t.forward(in_, nlevels, include_scale)

    # At level 1, the lowpass output will be the same size as the input. At
    # levels above that, it will be half the size per level
    extent = 512 * 2**(-(nlevels-1))
    assert p.lowpass_op.get_shape().as_list() == [1, extent, extent]
    assert p.lowpass_op.dtype == tf.float32

    for i in range(nlevels):
        extent = 512 * 2**(-(i+1))
        assert (p.highpasses_ops[i].get_shape().as_list() ==
                [1, extent, extent, 6])
        assert (p.highpasses_ops[i].dtype ==
                tf.complex64)
        if include_scale:
            assert (p.scales_ops[i].get_shape().as_list() ==
                    [1, 2*extent, 2*extent])
            assert p.scales_ops[i].dtype == tf.float32


@skip_if_no_tf
@pytest.mark.parametrize("nlevels, include_scale", [
    (2,False),
    (2,True),
    (4,False),
    (3,True)
])
def test_apply_reshaping(nlevels, include_scale):
    # Test the reshaping function of the Pyramid_tf class. This should apply
    # the same tf op to all of its operations. A good example would be to
    # remove the batch dimension from each op.
    in_ = tf.placeholder(tf.float32, [512, 512])
    t = Transform2d()
    # Calling forward with a 2d input will throw a warning
    p = t.forward(in_, nlevels, include_scale)
    f = lambda x: tf.squeeze(x, squeeze_dims=0)
    p.apply_reshaping(f)

    # At level 1, the lowpass output will be the same size as the input. At
    # levels above that, it will be half the size per level
    extent = 512 * 2**(-(nlevels-1))
    assert p.lowpass_op.get_shape().as_list() == [extent, extent]
    assert p.lowpass_op.dtype == tf.float32

    for i in range(nlevels):
        extent = 512 * 2**(-(i+1))
        assert p.highpasses_ops[i].get_shape().as_list() == [extent, extent, 6]
        assert p.highpasses_ops[i].dtype == tf.complex64
        if include_scale:
            assert p.scales_ops[i].get_shape().as_list() == [2*extent, 2*extent]
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
    t = Transform2d()
    # Calling forward with a 2d input will throw a warning
    Yl, Yh, Yscale = t.forward(in_, nlevels, include_scale, return_tuple=True)

    # At level 1, the lowpass output will be the same size as the input. At
    # levels above that, it will be half the size per level
    extent = 512 * 2**(-(nlevels-1))
    assert Yl.get_shape().as_list() == [1, extent, extent]
    assert Yl.dtype == tf.float32

    for i in range(nlevels):
        extent = 512 * 2**(-(i+1))
        assert Yh[i].get_shape().as_list() == [1, extent, extent, 6]
        assert Yh[i].dtype == tf.complex64
        if include_scale:
            assert Yscale[i].get_shape().as_list() == [1, 2*extent, 2*extent]
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
    t = Transform2d()
    p = t.forward(in_, nlevels, include_scale)

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
    t = Transform2d()

    Yl, Yh, Yscale = t.forward(in_, nlevels, include_scale, return_tuple=True)

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
    t = Transform2d()
    Yl, Yh, Yscale = t.forward_channels(in_, nlevels)

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
