.. figure:: https://github.com/pyqg/pyqg/blob/master/docs/_static/vortex_rollup.png
   :alt:


pyqg: Python Quasigeostrophic Model
===================================

|DOI| |conda| |pypi| |Build Status| |codecov| |docs| |binder|

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

Links
-----

-  HTML documentation: http://pyqg.readthedocs.org
-  Issue tracker: http://github.com/pyqg/pyqg/issues
-  Source code: http://github.com/pyqg/pyqg
-  pyfftw: http://github.com/hgomersall/pyFFTW

.. |DOI| image:: https://zenodo.org/badge/14957/pyqg/pyqg.svg
   :target: https://zenodo.org/badge/latestdoi/14957/pyqg/pyqg
.. |Build Status| image:: https://github.com/pyqg/pyqg/actions/workflows/ci.yaml/badge.svg
   :target: https://github.com/pyqg/pyqg/actions/workflows/ci.yaml
.. |codecov| image:: https://codecov.io/github/pyqg/pyqg/coverage.svg?branch=master
   :target: https://codecov.io/github/pyqg/pyqg?branch=master
.. |pypi| image:: https://badge.fury.io/py/pyqg.svg
   :target: https://badge.fury.io/py/pyqg
.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/pyqg.svg
   :target: https://anaconda.org/conda-forge/pyqg
.. |landscape| image:: https://landscape.io/github/pyqg/pyqg/master/landscape.svg?style=flat
   :target: https://landscape.io/github/pyqg/pyqg/master
   :alt: Code Health
.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyqg/pyqg/HEAD
.. |docs| image:: http://readthedocs.org/projects/pyqg/badge/?version=stable
   :target: http://pyqg.readthedocs.org/en/stable/?badge=stable
   :alt: Documentation Status
