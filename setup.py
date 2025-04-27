from setuptools import setup, Extension
from Cython.Build import cythonize
import warnings
import numpy as np

# This hack tells cython whether pyfftw is present
use_pyfftw_file = 'pyqg/.compile_time_use_pyfftw.pxi'
with open(use_pyfftw_file, 'wb') as f:
    try:
        import pyfftw
        f.write(b'DEF PYQG_USE_PYFFTW = 1')
    except ImportError:
        f.write(b'DEF PYQG_USE_PYFFTW = 0')
        warnings.warn('Could not import pyfftw. Model may be slower.')

ext_module = Extension(
    "pyqg.kernel",
    ["pyqg/kernel.pyx"],
    include_dirs = [np.get_include()],
)


setup(
    ext_modules = cythonize(ext_module),
)
