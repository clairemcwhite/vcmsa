from setuptools import setup
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

import unittest
def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='*_test.py')
    return test_suite

setup(name='vcmsa',
      version='0.1.0',
      description='Python library to perform vcmsa',
      url='http://github.com/clairemcwhite/vcmsa',
      author='Claire D. McWhite',
      author_email='cmcwhite@princeton.edu',
      license='MIT',
      scripts=['bin/vcmsa'],
      package_dir={'vcmsa':'src'},
      packages=['vcmsa'],
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
      include_package_data=True,
      test_suite='setup.my_test_suite'

     )
