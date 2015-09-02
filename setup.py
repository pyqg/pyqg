from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

VERSION='0.1.0'

DISTNAME='pyqg'
URL='http://github.com/pyqg/pyqg',
AUTHOR='pyqg team',
AUTHOR_EMAIL='pyqg-dev@googlegroups.com',
LICENSE='GPLv3',

DESCRIPTION='python quasigeostrophic model'
LONG_DESCRIPTION="""
pyqg is a python solver for quasigeostrophic systems. Quasigeostophic
equations are an approximation to the full fluid equations of motion in
the limit of strong rotation and stratitifcation and are most applicable
to geophysical fluid dynamics problems.

Students and researchers in ocean and atmospheric dynamics are the intended
audience of pyqg. The model is simple enough to be used by students new to
the field yet powerful enough for research. We strive for clear documentation
and thorough testing.

pyqg supports a variety of different configurations using the same
computational kernel. The different configurations are evolving and are
described in detail in the documentation. The kernel, implement in cython,
uses a pseudo-spectral method which is heavily dependent of the fast Fourier
transform. For this reason, pyqg depends on pyfftw and the FFTW Fourier
Transform library. The kernel is multi-threaded but does not support mpi.
Optimal performance will be achieved on a single system with many cores.

Links
-----

- HTML documentation: http://pyqg.readthedocs.org
- Issue tracker: http://github.com/pyqg/pyqg/issues
- Source code: http://github.com/pyqg/pyqg
"""

install_requires = [
    'cython',
    'numpy',
    'pyfftw'
]

# reathedocs can't and shouldn't build pyfftw
# apparently setup.py overrides docs/requirements.txt
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.remove('pyfftw')

tests_require = ['nose']

def readme():
    with open('README.md') as f:
        return f.read()

ext_module = Extension(
    "pyqg/kernel",
    ["pyqg/kernel.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Atmospheric Science'
]

setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      url=URL,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      packages=['pyqg'],
      install_requires=install_requires,
      ext_modules = cythonize(ext_module),
      include_dirs = [np.get_include()],
      tests_require = tests_require,
      test_suite = 'nose.collector',
      zip_safe=False)
