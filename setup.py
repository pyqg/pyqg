from setuptools import setup, Extension
from Cython.Build import cythonize
import warnings
import numpy as np
import os
import tempfile, subprocess, shutil

DISTNAME='pyqg'
URL='http://github.com/pyqg/pyqg'
AUTHOR='pyqg team'
AUTHOR_EMAIL='pyqg-dev@googlegroups.com'
LICENSE='MIT'

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

- HTML documentation: https://pyqg.readthedocs.io
- Issue tracker: https://github.com/pyqg/pyqg/issues
- Source code: https://github.com/pyqg/pyqg
"""

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Atmospheric Science'
]


### Dependency section ###
install_requires = [
    'cython',
    'numpy',
    'pyfftw'
]

# This hack tells cython whether pyfftw is present
use_pyfftw_file = 'pyqg/.compile_time_use_pyfftw.pxi'
with open(use_pyfftw_file, 'wb') as f:
    try:
        import pyfftw
        f.write(b'DEF PYQG_USE_PYFFTW = 1')
    except ImportError:
        f.write(b'DEF PYQG_USE_PYFFTW = 0')
        warnings.warn('Could not import pyfftw. Model may be slower.')

# check for openmp following
# http://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
# see http://openmp.org/wp/openmp-compilers/
omp_test = \
br"""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
"""

# python 3 needs rb

def check_for_openmp():
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'test.c'
    try:
        cc = os.environ['CC']
    except KeyError:
        cc = 'gcc'
    with open(filename, 'wb', 0) as file:
        file.write(omp_test)
    with open(os.devnull, 'wb') as fnull:
        try:
            result = subprocess.call([cc, '-fopenmp', filename],
                                     stdout=fnull, stderr=fnull)
        except FileNotFoundError:
            result = 1
    print('check_for_openmp() result: ', result)
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)

    return result==0

extra_compile_args = []
extra_link_args = []

use_openmp = True
if check_for_openmp() and use_openmp:
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
else:
    warnings.warn('Could not link with openmp. Model will be slow.')

# reathedocs can't and shouldn't build pyfftw
# apparently setup.py overrides docs/requirements.txt
#on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
#if on_rtd:
#    install_requires.remove('pyfftw')

tests_require = ['pytest']

def readme():
    with open('README.md') as f:
        return f.read()

def local_scheme(version):
    """Skip the local version (eg. +xyz of 0.6.1.dev4+gdf99fe2)
    to be able to upload to Test PyPI"""
    return ""

ext_module = Extension(
    "pyqg.kernel",
    ["pyqg/kernel.pyx"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(name=DISTNAME,
      description=DESCRIPTION,
      classifiers=CLASSIFIERS,
      long_description=LONG_DESCRIPTION,
      url=URL,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      packages=['pyqg'],
      install_requires=install_requires,
      setup_requires=["setuptools_scm"],
      use_scm_version={"local_scheme": local_scheme},
      python_requires='>=3.6',
      ext_modules = cythonize(ext_module),
      include_dirs = [np.get_include()],
      tests_require = tests_require,
      test_suite = 'nose.collector',
      zip_safe=False)
