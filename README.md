<p align="center">
    <img src="https://raw.githubusercontent.com/pyqg/pyqg/master/docs/_static/vortex_rollup.png">
</p>

# pyqg: Python Quasigeostrophic Model

[![Zenodo DOI](https://zenodo.org/badge/14957/pyqg/pyqg.svg)][doi]
[![PyQG on conda-forge](https://img.shields.io/conda/vn/conda-forge/pyqg)][conda]
[![PyQG on PyPI](https://img.shields.io/pypi/v/pyqg)][pypi]
[![Build Status](https://github.com/pyqg/pyqg/actions/workflows/ci.yaml/badge.svg)][buildstatus]
[![Codecov](https://codecov.io/github/pyqg/pyqg/coverage.svg?branch=master)][codecov]
[![Documentation](https://readthedocs.org/projects/pyqg/badge/?version=stable)][docs]
[![Binder](https://mybinder.org/badge_logo.svg)][binder]

pyqg is a python solver for quasigeostrophic systems. Quasigeostophic
equations are an approximation to the full fluid equations of motion in
the limit of strong rotation and stratitifcation and are most applicable
to geophysical fluid dynamics problems.

Students and researchers in ocean and atmospheric dynamics are the
intended audience of pyqg. The model is simple enough to be used by
students new to the field yet powerful enough for research. We strive
for clear documentation and thorough testing.

pyqg supports a variety of different configurations using the same
computational kernel. The different configurations are evolving and are
described in detail in the documentation. The kernel, implement in
cython, uses a pseudo-spectral method which is heavily dependent on the
fast Fourier transform. For this reason, pyqg tries to use pyfftw and
the FFTW Fourier Transform library. (If pyfftw is not available, it
falls back on numpy.fft) With pyfftw, the kernel is multi-threaded but
does not support mpi. Optimal performance will be achieved on a single
system with many cores.

## Links
-  HTML documentation: https://pyqg.readthedocs.org
-  Issue tracker: https://github.com/pyqg/pyqg/issues
-  Source code: https://github.com/pyqg/pyqg
-  pyfftw: https://github.com/pyFFTW/pyFFTW

[doi]: https://zenodo.org/badge/latestdoi/14957/pyqg/pyqg
[conda]: https://anaconda.org/conda-forge/pyqg
[pypi]: https://pypi.org/project/pyqg/
[buildstatus]: https://github.com/pyqg/pyqg/actions/workflows/ci.yaml
[codecov]: https://app.codecov.io/github/pyqg/pyqg/branch/master
[docs]: https://pyqg.readthedocs.org
[binder]: https://mybinder.org/v2/gh/pyqg/pyqg/HEAD
