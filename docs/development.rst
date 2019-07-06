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

    $ conda create --yes -n test_env python=2.7 pip nose numpy cython scipy nose
    $ conda install --yes -n test_env -c nanshe pyfftw
    $ source activate test_env
    $ pip install pyqg
    $ python -c 'import pyqg; print(pyqg.__version__);'
    $ source deactivate
    $ conda env remove --yes -n test_env

Release Procedure
=================

Once we are ready for a new release, someone needs to make a pull request which
updates the version number in setup.py. Also make sure that whats-new.rst in
the docs is up to date.

After the new version number PR has been merged, create a new `release`_ in
github.

The step of publishing to `pypi`_ has to be done manually from the command line.
(Note: I figured out how this works from these `instructions`_).
After the new release has been created, do the following.

.. code-block:: bash

    $ cd pyqg
    $ git fetch upstream
    $ git checkout master
    $ git rebease upstream/master
    # test the release before publishing
    $ python setup.py register -r pypitest
    $ python setup.py sdist upload -r pypitest
    # if that goes well, publish it
    $ python setup.py register -r pypi
    $ python setup.py sdist upload -r pypi

Note that pypi will not let you publish the same release twice, so make sure
you get it right!

.. _pypi: https://pypi.python.org/pypi/pyqg
.. _release: https://help.github.com/articles/creating-releases/
.. _instructions: http://peterdowns.com/posts/first-time-with-pypi.html
