import numpy as np

from dtcwt import reflect

def setup():
    global ramp, reflected

    # Create a simple linear ramp and reflect it
    ramp = np.linspace(-100, 100, 500)
    reflected = reflect(ramp, 30, 40)

def test_linear_ramp_boundaries():
    # Check boundaries
    assert not np.any(reflected < 30)
    assert not np.any(reflected > 40)

def test_linear_ramp_values():
    # Check that valid region is unchanged
    r = np.logical_and(ramp >= 30, ramp <= 40)
    assert np.any(r)
    assert np.all(reflected[r] == ramp[r])

def test_non_array_input():
    ramp = np.linspace(-100, 100, 500).tolist()
    reflected = reflect(ramp, 30, 40)

    # Check boundaries
    assert not np.any(reflected < 30)
    assert not np.any(reflected > 40)


# vim:sw=4:sts=4:et
