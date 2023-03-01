from setuptools import setup, Extension
from Cython.Build import cythonize
import warnings
import numpy as np
import os
import tempfile, subprocess, shutil

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

ext_module = Extension(
    "pyqg.kernel",
    ["pyqg/kernel.pyx"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs = [np.get_include()],
)

setup(
    ext_modules = cythonize(ext_module),
)
