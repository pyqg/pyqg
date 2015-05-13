.. pyqg documentation master file, created by
   sphinx-quickstart on Sun May 10 14:44:32 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. toctree::
   :maxdepth: 2
    
    preface


Welcome to pyqg's documentation!
================================

The PyQG team aims to build a "git generation" community quasi-geostrophic model in Python. Ideally, PyQG will be a tool that is easy-to-use, high-level and unit-tested.

Please note the following about this project:

* PyQG is in its birth, and its a side-project of its core developers.
* ...

Installation
-------------
PyQG assumes you have python installed on your computer. The only strict requirement is *numpy*, but it is convenient to install *matplotlib*, *scipy* and *mkl*, and some of the examples assume you have those packages. We strongly encourage you to get python from pre-cooked distributions such as `anaconda <https://store.continuum.io/cshop/anaconda/>`_ and `canopy <https://www.enthought.com/products/canopy/>`_

To speed-up calculations, you can install

* pyfftw
* ...

These are easily installed in anaconda or canopy. For example

.. code-block:: bash

    $ conda install pyfftw


You can download `PyQG <https://github.com/rabernat/pyqg>`_ from its repository. If you use git, you can simply clone the repo:

.. code-block:: bash

    $ git clone https://github.com/rabernat/pyqg.git

You should install PyQG on your system:

.. code-block:: bash

    $ python setup.py install

If you want to make changes in the code, set up the development mode:

.. code-block:: bash

    $ python setup.py develop


