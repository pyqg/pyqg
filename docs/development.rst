Development
###########

Team
====

- `Malte Jansen`_, University of Chicago

- `Ryan Abernathey`_, Columbia University / LDEO

- `Cesar Rocha`_, Woods Hole Oceanographic Institution

- `Francis Poulin`_, University of Waterloo

.. _Malte Jansen: http://geosci.uchicago.edu/people/malte-jansen/
.. _Ryan Abernathey: http://rabernat.github.io
.. _Cesar Rocha: http://www.cbrocha.com
.. _Francis Poulin: https://uwaterloo.ca/poulin-research-group/

History
=======

The numerical approach of pyqg was originally inspired by a MATLAB code by
`Glenn Flierl`_ of MIT, who was a teacher and mentor to Ryan and Malte.
It would be hard to find anyone in the world who knows more about this sort
of model than Glenn. Malte implemented a python version of the two-layer
model while at GFDL. In the summer of 2014, while both were at the `WHOI GFD
Summer School`_, Ryan worked with Malte refactor the code into a proper
python package. Cesar got involved and brought pyfftw into the project. Ryan
implemented a cython kernel. Cesar and Francis implemented the barotropic and
sqg models.

.. _WHOI GFD Summer School: https://www.whoi.edu/gfd/
.. _Glenn Flierl: https://eapsweb.mit.edu/people/grflierl

Future
======

By adopting open-source best practices, we hope pyqg will grow into a widely
used, community-based project. We know that many other research groups have
their own "in house" QG models. You can get involved by trying out the model,
filing issues_ if you find problems, and making `pull requests`_ if you make
improvements.

.. _issues: https://github.com/pyqg/pyqg/issues
.. _pull requests: https://github.com/pyqg/pyqg/pulls

.. _dev-workflow:

Develpment Workflow
===================

Anyone interested in helping to develop pyqg needs to create their own fork
of our `git repository`. (Follow the github `forking instructions`_. You
will need a github account.)

.. _git repository: https://github.com/pyqg/pyqg
.. _forking instructions: https://help.github.com/articles/fork-a-repo/

Clone your fork on your local machine.

.. code-block:: bash

    $ git clone git@github.com:USERNAME/pyqg

(In the above, replace USERNAME with your github user name.)

Then set your fork to track the upstream pyqg repo.

.. code-block:: bash

    $ cd pyqg
    $ git remote add upstream git://github.com/pyqg/pyqg.git

You will want to periodically sync your master branch with the upstream master.

.. code-block:: bash

    $ git fetch upstream
    $ git rebase upstream/master

Never make any commits on your local master branch. Instead open a feature
branch for every new development task.

.. code-block:: bash

    $ git checkout -b cool_new_feature

(Replace `cool_new_feature` with an appropriate description of your feature.)
At this point you work on your new feature, using `git add` to add your
changes. When your feature is complete and well tested, commit your changes

.. code-block:: bash

    $ git commit -m 'did a bunch of great work'

and push your branch to github.

.. code-block:: bash

    $ git push origin cool_new_feature

At this point, you go find your fork on github.com and create a `pull
request`_. Clearly describe what you have done in the comments. If your
pull request fixes an issue or adds a useful new feature, the team will
gladly merge it.

.. _pull request: https://help.github.com/articles/using-pull-requests/

After your pull request is merged, you can switch back to the master branch,
rebase, and delete your feature branch. You will find your new feature
incorporated into pyqg.

.. code-block:: bash

    $ git checkout master
    $ git fetch upstream
    $ git rebase upstream/master
    $ git branch -d cool_new_feature

Virtual Environment
===================

This is how to create a virtual environment into which to test-install pyqg,
install it, check the version, and tear down the virtual environment.

.. code-block:: bash

    $ conda create --yes -n test_env python=3.9 pip nose numpy cython scipy nose
    $ conda install --yes -n test_env -c conda-forge pyfftw
    $ source activate test_env
    $ pip install pyqg
    $ python -c 'import pyqg; print(pyqg.__version__);'
    $ conda deactivate
    $ conda env remove --yes -n test_env

Release Procedure
=================

Once we are ready for a new release, someone needs to make a pull request which
updates `docs/whats-new.rst` in preparation for the new version.  Then, you can
simply create a new `release`_ in Github, adding a new tag for the new version
(following `semver`_) and clicking "Auto-generate release notes" to summarize
changes since the last release (with further elaboration if necessary).

After the release is created, a new version should be published to `pypi`_
automatically.

However, before creating the release, it's worth checking `testpypi`_ to ensure
the new version works. You can do that by:

#. Verifying the `most recent test publish`_ succeeded (and is for the most
   recent commit)

#. Finding the corresponding pre-release version in pyqg's `TestPyPI history`_
   (should look like `X.Y.Z.devN`)

#. Installing that version locally as follows:

.. code-block:: bash

    # Create a temporary directory with a fresh conda environment
    $ mkdir ~/tmp
    $ cd ~/tmp
    $ conda create --yes -n test_env python=3.9 pip nose numpy cython scipy nose setuptools setuptools_scm
    $ source activate test_env
    $ pip install pyfftw # or install with conda-forge
    
    # Install the latest pre-release version of pyqg
    $ pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-cache-dir pyqg==X.Y.Z.devN

    # Ensure this imports successfully and prints out the pre-release version (X.Y.Z.devN)
    $ python -c 'import pyqg; print(pyqg.__version__);'

    # Clean up and remove the test environment
    $ conda deactivate
    $ conda env remove --yes -n test_env

If this all works, then you should be ready to create the Github `release`_.

.. _testpypi: https://packaging.python.org/en/latest/guides/using-testpypi
.. _pypi: https://pypi.python.org/pypi/pyqg
.. _release: https://help.github.com/articles/creating-releases/
.. _instructions: http://peterdowns.com/posts/first-time-with-pypi.html
.. _semver: https://semver.org/
.. _most recent test publish: https://github.com/pyqg/pyqg/actions/workflows/publish-to-test-pypi.yml
.. _TestPyPI history: https://test.pypi.org/project/pyqg/#history
