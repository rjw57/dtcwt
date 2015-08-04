import dtcwt
import dtcwt.numpy as npbackend
import dtcwt.opencl as clbackend

from unittest import TestCase
from pytest import raises
from .util import skip_if_no_cl

class TestSwitchBackends(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSwitchBackends, self).__init__(*args, **kwargs)
        self._orig_stack = None

    def setUp(self):
        # Preserve _BACKEND_STACK in case test fails. This stops one test
        # failure messing up other tests. This requires knowledge of how dtcw
        # manages the backend stack which is not part of the public API.
        #
        # Perhaps we need to add __enter__() and __exit__() functions to a
        # stack object.
        self._orig_stack = list(dtcwt._BACKEND_STACK)

    def tearDown(self):
        dtcwt._BACKEND_STACK = self._orig_stack

    def test_default_backend(self):
        assert dtcwt.Transform2d is npbackend.Transform2d
        assert dtcwt.Pyramid is npbackend.Pyramid
        assert dtcwt.backend_name == 'numpy'

    @skip_if_no_cl
    def test_switch_to_opencl(self):
        assert dtcwt.Transform2d is npbackend.Transform2d
        assert dtcwt.Pyramid is npbackend.Pyramid
        assert dtcwt.backend_name == 'numpy'
        dtcwt.push_backend('opencl')
        assert dtcwt.Transform2d is clbackend.Transform2d
        assert dtcwt.Pyramid is clbackend.Pyramid
        assert dtcwt.backend_name == 'opencl'
        dtcwt.pop_backend()
        assert dtcwt.Transform2d is npbackend.Transform2d
        assert dtcwt.Pyramid is npbackend.Pyramid
        assert dtcwt.backend_name == 'numpy'

    def test_switch_to_numpy(self):
        assert dtcwt.Transform2d is npbackend.Transform2d
        assert dtcwt.Pyramid is npbackend.Pyramid
        assert dtcwt.backend_name == 'numpy'
        dtcwt.push_backend('numpy')
        assert dtcwt.Transform2d is npbackend.Transform2d
        assert dtcwt.Pyramid is npbackend.Pyramid
        assert dtcwt.backend_name == 'numpy'
        dtcwt.pop_backend()
        assert dtcwt.Transform2d is npbackend.Transform2d
        assert dtcwt.Pyramid is npbackend.Pyramid
        assert dtcwt.backend_name == 'numpy'

    def test_switch_to_invalid(self):
        with raises(ValueError):
            dtcwt.push_backend('does-not-exist')

    def test_no_pop_default_backend(self):
        with raises(IndexError):
            dtcwt.pop_backend()

def test_backend_with_guard():
    """Test that manipulating the stack with preserve_backend_stack() will
    return the stack to its pristine state even i pushes and pops are
    imbalanced.

    """
    assert len(dtcwt._BACKEND_STACK) == 1
    with dtcwt.preserve_backend_stack():
        dtcwt.push_backend('numpy')
        assert len(dtcwt._BACKEND_STACK) == 2
    assert len(dtcwt._BACKEND_STACK) == 1

def test_backend_with_guard_and_exception():
    """Tests that even when preserving the backend stack, an exception is
    correctly propagated.

    """
    dtcwt.push_backend('numpy')
    assert len(dtcwt._BACKEND_STACK) == 2
    assert dtcwt.backend_name == 'numpy'
    with raises(RuntimeError):
        with dtcwt.preserve_backend_stack():
            dtcwt.push_backend('opencl')
            assert dtcwt.backend_name == 'opencl'
            assert len(dtcwt._BACKEND_STACK) == 3
            raise RuntimeError('test error')
    assert dtcwt.backend_name == 'numpy'
    assert len(dtcwt._BACKEND_STACK) == 2
