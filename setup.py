from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

def readme():
    with open('README.md') as f:
        return f.read()

ext_module = Extension(
    "pyqg/kernel",
    ["pyqg/kernel.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)


setup(name='pyqg',
      version='0.1',
      description='A pure python quasigeostrophic model',
      url='http://github.com/rabernat/pyqg',
      author='Ryan Abernathey & Malte Jansen',
      author_email='rpa@ldeo.columbia.edu',
      license='MIT',
      packages=['pyqg'],
      install_requires=[
          'numpy',
      ],
      #ext_modules = cythonize("pyqg/*.pyx"),
      ext_modules = cythonize(ext_module),
      include_dirs = [np.get_include()],
      test_suite = 'nose.collector',
      zip_safe=False)
