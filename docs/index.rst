.. pyqg documentation master file, created by
   sphinx-quickstart on Sun May 10 14:44:32 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/vortex_rollup.png
   :align: center

pyqg: Python Quasigeostrophic Model
===================================

pyqg is a python solver for quasigeostrophic systems. Quasigeostophic
equations are an approximation to the full fluid equations of motion in
the limit of strong rotation and stratification and are most applicable
to geophysical fluid dynamics problems.

Students and researchers in ocean and atmospheric dynamics are the intended
audience of pyqg. The model is simple enough to be used by students new to
the field yet powerful enough for research. We strive for clear documentation
and thorough testing.

pyqg supports a variety of different configurations using the same
computational kernel. The different configurations are evolving and are
described in detail in the documentation. The kernel, implement in cython,
uses a pseudo-spectral method which is heavily dependent of the fast Fourier
transform. For this reason, pyqg tries to use pyfftw_ and the FFTW_ Fourier
Transform library. (If pyfftw is not available, it falls back on numpy.fft)
With pyfftw, the kernel is multi-threaded but does not support mpi. Optimal
performance will be achieved on a single system with many cores.

.. _FFTW: http://www.fftw.org/
.. _pyfftw: http://github.com/hgomersall/pyFFTW

Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   equations    
   examples
   api
   development
   whats-new
