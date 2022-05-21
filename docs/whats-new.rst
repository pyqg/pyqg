What's New
==========

v0.7.2 (19 May 2022)
--------------------
- Temporarily removes subgrid forcing method

v0.7.1 (17 May 2022)
--------------------
- Fixes packaging bug

v0.7.0 (16 May 2022)
--------------------
- Allow parameterizations as first-class objects
- Add an initial library of parameterizations
- Add tools for comparing diagnostics
- Add a method for computing subgrid forcing

v0.6.0 (16 May 2022)
--------------------
- Generalize definition of parameterization spectrum diagnostic
- Add enstrophy budget diagnostics
- Normalize and unitize all diagnostics
- Fix issues with calculating isotropic spectra
- Other refactors and bug fixes

v0.5.0 (23 Mar. 2022)
---------------------
- Added support for online parameterizations
- Dropped support for Python 2.7
- Improvements to the development and release process
- Miscellaneous bug fixes

v0.4.0 (15 Sep. 2021)
---------------------
- Refactored diagnostics
- Added xarray support
- Improvements to documentation and build process
- Miscellaneous bug fixes

v0.3.0 (23 Nov. 2019)
---------------------
- Revived development after long hiatus
- Reverted some experimental changes
- Several small bug fixes and documentation corrections
- Updated CI and doc build environments
- Adopted to versioneer for package versioning

v0.2.0 (27 April 2016)
----------------------

Added compatibility with python 3.

Implemented linear baroclinic stability analysis method.

Implemented vertical mode methods and modal KE and PE spectra diagnostics.

Implemented multi-layer subclass.

Added new logger that leverages on built-in python logging.

Changed license to MIT.

v0.1.4 (22 Oct 2015)
--------------------

Fixed bug related to the sign of advection terms (:issue:`86`).

Fixed bug in _calc_diagnostics (:issue:`75`). Now diagnostics start being averaged at tavestart.

v0.1.3 (4 Sept 2015)
--------------------

Fixed bug in setup.py that caused openmp check to not work.

v0.1.2 (2 Sept 2015)
--------------------

Package was not building properly through pip/pypi. Made some tiny changes to
setup script. pypi forces you to increment the version number.

v0.1.1 (2 Sept 2015)
--------------------

A bug-fix release with no api or feature changes. The kernel has been modified
to support numpy fft routines.

- Removed pyfftw depenency (:issue:`53`)
- Cleaning of examples

v0.1 (1 Sept 2015)
------------------

Initial release.
