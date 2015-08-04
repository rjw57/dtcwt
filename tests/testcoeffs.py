from dtcwt.coeffs import biort, qshift

from pytest import raises

def test_antonini():
    h0o, g0o, h1o, g1o = biort('antonini')
    assert h0o.shape[0] == 9
    assert g0o.shape[0] == 7
    assert h1o.shape[0] == 7
    assert g1o.shape[0] == 9

def test_legall():
    h0o, g0o, h1o, g1o = biort('legall')
    assert h0o.shape[0] == 5
    assert g0o.shape[0] == 3
    assert h1o.shape[0] == 3
    assert g1o.shape[0] == 5

def test_near_sym_a():
    h0o, g0o, h1o, g1o = biort('near_sym_a')
    assert h0o.shape[0] == 5
    assert g0o.shape[0] == 7
    assert h1o.shape[0] == 7
    assert g1o.shape[0] == 5

def test_near_sym_a():
    h0o, g0o, h1o, g1o = biort('near_sym_b')
    assert h0o.shape[0] == 13
    assert g0o.shape[0] == 19
    assert h1o.shape[0] == 19
    assert g1o.shape[0] == 13

def test_qshift_06():
    coeffs = qshift('qshift_06')
    assert len(coeffs) == 8
    for v in coeffs:
        assert v.shape[0] == 10

def test_qshift_a():
    coeffs = qshift('qshift_a')
    assert len(coeffs) == 8
    for v in coeffs:
        assert v.shape[0] == 10

def test_qshift_b():
    coeffs = qshift('qshift_b')
    assert len(coeffs) == 8
    for v in coeffs:
        assert v.shape[0] == 14

def test_qshift_c():
    coeffs = qshift('qshift_c')
    assert len(coeffs) == 8
    for v in coeffs:
        assert v.shape[0] == 16

def test_qshift_d():
    coeffs = qshift('qshift_d')
    assert len(coeffs) == 8
    for v in coeffs:
        assert v.shape[0] == 18

def test_non_exist_biort():
    with raises(IOError):
        biort('this-does-not-exist')

def test_non_exist_qshift():
    with raises(IOError):
        qshift('this-does-not-exist')

def test_wrong_type_a():
    with raises(ValueError):
        biort('qshift_06')

def test_wrong_type_b():
    with raises(ValueError):
        qshift('antonini')

# vim:sw=4:sts=4:et
