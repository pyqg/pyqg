.. _installation:

Installation
############

Requirements
============

- Python 2.7. (Python 3 support is in the works) 
- `numpy <http://www.numpy.org/>`__ (1.6 or later)
- fftw_ (3.3 or later)
- pyfftw_ (0.9.2 or later)

.. _fftw: http://www.fftw.org/
.. _pyfftw: http://github.com/hgomersall/pyFFTW

Instructions
============

In our opinion, the best way to get python and numpy is to use a distribution
such as Anaconda_ (recommended) or Canopy_. These provide robust package
management and come with many other useful packages for scientific computing.
The pyqg developers are mostly using anaconda.

Because pyqg is a pseudo-spectral code, it realies heavily on fast-Fourier
transforms (FFTs), which are the main performance bottlneck. For this reason, we
have made fftw_ (a fast, multithreaded, open source C library) and pyfftw_ (a
python wrapper around fftw) core dependencies for pyqg. Installing fftw and
pyfftw can be slightly painful. Hopefully the instructions below are sufficient.
If not, please `send feedback <http://github.com/pyqg/pyqg/issues>`__.

.. _Anaconda: https://store.continuum.io/cshop/anaconda
.. _Canopy: https://www.enthought.com/products/canopy

.. note::
    We would like to add compatibilty for numpy.fft in the near future,
    eliminating the dependency on fftw and pyfftw.

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
The `mforbes <https://anaconda.org/mforbes>`__ channel version was selected
for this documentation because its pyfftw package is compatible with the
latest version of numpy (1.9.2) and both linux and mac platforms. We don't know
who mforbes is, but we are greatful to him/her.

To install pyfftw from the mforbes channel, open a terminal and run
the command

.. code-block:: bash

    $ conda install -c mforbes pyfftw

If this doesn't work for you, or if it asks you to upgrade / downgrade more of
your core pacakges (e.g. numpy) than you would like, you can easily try
replacing ``mforbes`` with one of the other `channels
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

With pyfftw installed, you can now install pyqg. The easiest way is with pip:

.. code-block:: bash

    $ pip install pyqg

You can also clone the `pyqg git repository <https://github.com/pyqg/pyqg`__ to
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
