.. _installation:

Installation
############

Requirements
============

The only requirements are

- Python 2.7. (Python 3 support is in the works)
- numpy_ (1.6 or later)
- Cython (0.2 or later)

Because pyqg is a pseudo-spectral code, it realies heavily on fast-Fourier
transforms (FFTs), which are the main performance bottlneck. For this reason,
we try to use fftw_ (a fast, multithreaded, open source C library) and pyfftw_
(a python wrapper around fftw). These packages are optional, but they are
strongly recommended for anyone doing high-resolution, numerically demanding
simulations.

- fftw_ (3.3 or later)
- pyfftw_ (0.9.2 or later)

If pyqg can't import pyfftw at compile time, it will fall back on numpy_'s fft
routines.

.. _numpy:  http://www.numpy.org/
.. _fftw: http://www.fftw.org/
.. _pyfftw: http://github.com/hgomersall/pyFFTW

Instructions
============

In our opinion, the best way to get python and numpy is to use a distribution
such as Anaconda_ (recommended) or Canopy_. These provide robust package
management and come with many other useful packages for scientific computing.
The pyqg developers are mostly using anaconda.

.. note::
    If you don't want to use pyfftw and are content with numpy's slower
    performance, you can skip ahead to :ref:`install-pyqg`.

Installing fftw and
pyfftw can be slightly painful. Hopefully the instructions below are sufficient.
If not, please `send feedback <http://github.com/pyqg/pyqg/issues>`__.

.. _Anaconda: https://store.continuum.io/cshop/anaconda
.. _Canopy: https://www.enthought.com/products/canopy

Installing fftw and pyfftw
--------------------------

Once you have installed pyfftw via one of these paths, you can proceed to
:ref:`install-pyqg`.

The easy way: installing with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using Anaconda_, we have discovered that you can easily install
pyffw using the ``conda`` command. Although pyfftw is not part of the `main
Anaconda distribution <http://docs.continuum.io/anaconda/pkg-docs>`__, it is
distributed as a conda pacakge through several `user channels
<https://anaconda.org/>`__.

There is a useful `blog post
<https://dranek.com/blog/2014/Feb/conda-binstar-and-fftw/>`__ describing how
the pyfftw conda package was created. There are currently 13
`pyfftw user packages <https://anaconda.org/search?q=pyfftw>`__
hosted on anaconda.org. Each has different dependencies and platform support
(e.g. linux, windows, mac.)
The `nanshe <https://anaconda.org/nanshe/pyfftw>`__ channel version is the most
popular and appears to have the broadest cross-platform support. We don't know
who nanshe is, but we are greatful to him/her.

To install pyfftw from the nanshe channel, open a terminal and run
the command

.. code-block:: bash

    $ conda install -c nanshe pyfftw

If this doesn't work for you, or if it asks you to upgrade / downgrade more of
your core pacakges (e.g. numpy) than you would like, you can easily try
replacing ``nanshe`` with one of the other `channels
<https://anaconda.org/search?q=pyfftw>`__.

The hard way: installing from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the most difficult step for new users. You will probably have to build
FFTW3 from source. However, if you are using Ubuntu linux, you can save yourself
some trouble by installing fftw using the apt package manager

.. code-block:: bash

    $ sudo apt-get install libfftw3-dev libfftw3-doc

Otherwise you have to build FFTW3 from source. Your main resource for the
`FFTW homepage <http://www.fftw.org/>`__. Below we summarize the steps

First `download <http://www.fftw.org/download.html>`__ the source code.

.. code-block:: bash

    $ wget http://www.fftw.org/fftw-3.3.4.tar.gz
    $ tar -xvzf fftw-3.3.4.tar.gz
    $ cd fftw-3.3.4

Then run the configure command

.. code-block:: bash

    $ ./configure --enable-threads --enable-shared

.. note::
    If you don't have root privileges on your computer (e.g. on a shared
    cluster) the best approach is to ask your system administrator to install
    FFTW3 for you. If that doesn't work, you will have to install the FFTW3
    libraries into a location in your home directory (e.g. ``$HOME/fftw``) and
    add the flag ``--prefix=$HOME/fftw`` to the configure command above.

Then build the software

.. code-block:: bash

    $ make

Then install the software

.. code-block:: bash

    $ sudo make install

This will install the FFTW3 libraries into you system's library directory.
If you don't have root privileges (see note above), remove the ``sudo``. This
will install the libraries into the ``prefix`` location you specified.

You are not done installing FFTW yet. pyfftw requires special versions
of the FFTW library specialized to different data types (32-bit floats and
double-long floars). You need to-configure and re-build FFTW two more times
with extra flags.

.. code-block:: bash

    $ ./configure --enable-threads --enable-shared --enable-float
    $ make
    $ sudo make install
    $ ./configure --enable-threads --enable-shared --enable-long-double
    $ make
    $ sudo make install

At this point, you FFTW installation is complete. We now move on to pyfftw.
pyfftw is a python wrapper around the FFTW libraries. The easiest way to
install it is using ``pip``:

.. code-block:: bash

    $ pip install pyfftw

or if you don't have root privileges

.. code-block:: bash

    $ pip install pyfftw --user

If this fails for some reason, you can manually download and install it
according to the `instructions on github
<https://github.com/hgomersall/pyFFTW#building>`__. First clone the repository:

.. code-block:: bash

    $ git clone https://github.com/hgomersall/pyFFTW.git

Then install it

.. code-block:: bash

    $ cd pyFFTW
    $ python setup.py install

or

.. code-block:: bash

    $ python setup.py install --user

if you don't have root privileges. If you installed FFTW in a non-standard
location (e.g. $HOME/fftw), you might have to do something tricky at this point
to make sure pyfftw can find FFTW. (I figured this out once, but I can't
remember how.)

.. _install-pyqg:

Installing pyqg
---------------
.. note::
    The pyqg kernel is written in Cython and uses OpenMP to parallelise some operations for a performance boost.
    If you are using Mac OSX Yosemite or later OpenMP support is not available out of the box.  While pyqg will
    still run without OpenMP, it will not be as fast as it can be. See :ref:`advanced-install` below for more
    information on installing on OSX with OpenMP support.

With pyfftw installed, you can now install pyqg. The easiest way is with pip:

.. code-block:: bash

    $ pip install pyqg

You can also clone the `pyqg git repository <https://github.com/pyqg/pyqg>`__ to
use the latest development version.

.. code-block:: bash

    $ git clone https://github.com/pyqg/pyqg.git

Then install pyqg on your system:

.. code-block:: bash

    $ python setup.py install [--user]

(The ``--user`` flag is optional--use it if you don't have root privileges.)

If you want to make changes in the code, set up the development mode:

.. code-block:: bash

    $ python setup.py develop

pyqg is a work in progress, and we really encourage users to contribute to its
:doc:`/development`


.. _advanced-install:

Installing with OpenMP support on OSX
-------------------------------------

There are two options for installing on OSX with OpenMP support.  Both methods require using the Anaconda distribution of
Python.

1. Using Homebrew

Install the GCC-5 compiler in ``/usr/local`` using Homebrew:

.. code-block:: bash

    $ brew install gcc --without-multilib --with-fortran

Install Cython from the conda repository

.. code-block:: bash

    $ conda install cython

Install pyqg using the homebrew ``gcc`` compiler

.. code-block:: bash

    $ CC=/usr/local/bin/gcc-5 pip install pyqg


2. Using the HPC precompiled gcc binaries.

The `HPC for Mac OSX <http://hpc.sourceforge.net/>`__ sourceforge project has copies of the latest ``gcc`` precompiled for Mac OSX.  Download the latest version of gcc from the HPC site and follow the installation instructions.

Install Cython from the conda repository

.. code-block:: bash

    $ conda install cython

Install pyqg using the HPC ``gcc`` compiler

.. code-block:: bash

    $ CC=/usr/local/bin/gcc pip install pyqg
