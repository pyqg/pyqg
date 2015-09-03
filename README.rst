.. figure:: https://github.com/pyqg/pyqg/blob/master/docs/_static/vortex_rollup.png
   :alt: 

pyqg: Python Quasigeostrophic Model
===================================

|DOI| |Build Status|

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
cython, uses a pseudo-spectral method which is heavily dependent of the
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
.. |Build Status| image:: https://travis-ci.org/pyqg/pyqg.svg?branch=master
   :target: https://travis-ci.org/pyqg/pyqg
