#!/usr/bin/env python

from __future__ import print_function, division

import os
import sys
import textwrap
import timeit

# Add project root to path
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    t = timeit.Timer(
        setup=textwrap.dedent('''
            import dtcwt
            import dtcwt.registration
            import tests.datasets as datasets

            print('Loading datasets...')
            f1, f2 = datasets.regframes('tennis')

            print('Transforming datasets...')
            transform = dtcwt.Transform2d()
            t1 = transform.forward(f1, nlevels=6)
            t2 = transform.forward(f2, nlevels=6)
        '''),
        stmt=textwrap.dedent('''
            print('Registering datasets...')
            reg = dtcwt.registration.estimatereg(t1, t2)
        ''')
    )

    number = 20
    try:
        secs = t.timeit(number)
        print('{0:.2f}s for {1} iterations => {2:.2f}s/iteration'.format(
                secs, number, secs / number
            ))
    except:
        t.print_exc()

if __name__ == '__main__':
    main()
