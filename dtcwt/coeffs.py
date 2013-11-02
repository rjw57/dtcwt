import os
from numpy import load

DATADIR = os.path.join(os.path.dirname(__file__), 'data')

COEFF_CACHE = {}

def _load_from_file(basename, varnames):
    filename = os.path.join(DATADIR, basename + '.npz')

    try:
        mat = COEFF_CACHE[filename]
    except KeyError:
        mat = load(filename)
        COEFF_CACHE[filename] = mat

    try:
        return tuple(mat[k] for k in varnames)
    except KeyError:
        raise ValueError('Wavelet does not define ({0}) coefficients'.format(', '.join(varnames)))

def biort(name):
    """Load level 1 wavelet by name.

    :param name: a string specifying the wavelet family name
    :returns: a tuple of vectors giving filter coefficients

    ============ ============================================
    Name         Wavelet
    ============ ============================================
    antonini     Antonini 9,7 tap filters.
    legall       LeGall 5,3 tap filters.
    near_sym_a   Near-Symmetric 5,7 tap filters.
    near_sym_b   Near-Symmetric 13,19 tap filters.
    near_sym_b_bp Near-Symmetric 13,19 tap filters + BP filter
    ============ ============================================

    Return a tuple whose elements are a vector specifying the h0o, g0o, h1o and
    g1o coefficients.

    :raises IOError: if name does not correspond to a set of wavelets known to the library.
    :raises ValueError: if name specifies a :py:func:`qshift` wavelet.

    """
    if name == 'near_sym_b_bp':
        return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o', 'h2o', 'g2o'))
    else:
        return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o'))

def qshift(name):
    """Load level >=2 wavelet by name,

    :param name: a string specifying the wavelet family name
    :returns: a tuple of vectors giving filter coefficients

    ============ ============================================
    Name         Wavelet
    ============ ============================================
    qshift_06    Quarter Sample Shift Orthogonal (Q-Shift) 10,10 tap filters, 
                 (only 6,6 non-zero taps).
    qshift_a     Q-shift 10,10 tap filters,
                 (with 10,10 non-zero taps, unlike qshift_06).
    qshift_b     Q-Shift 14,14 tap filters.
    qshift_c     Q-Shift 16,16 tap filters.
    qshift_d     Q-Shift 18,18 tap filters.
    qshift_b_bp  Q-Shift 18,18 tap filters + BP
    ============ ============================================

    Return a tuple whose elements are a vector specifying the h0a, h0b, g0a,
    g0b, h1a, h1b, g1a and g1b coefficients.

    :raises IOError: if name does not correspond to a set of wavelets known to the library.
    :raises ValueError: if name specifies a :py:func:`biort` wavelet.

    """
    if name == 'qshift_b_bp':
        return _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b', 'g1a', 'g1b','h2a', 'h2b'))
    else:
        return _load_from_file(name, ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b', 'g1a', 'g1b'))

# vim:sw=4:sts=4:et
