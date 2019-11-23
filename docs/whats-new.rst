What's New
==========

v0.3.1 (unreleased)
-------------------


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
