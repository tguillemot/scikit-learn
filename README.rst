.. -*- mode: rst -*-

About
=====

scikits.learn is a python module for machine learning built on top of
scipy.

The project was started in 2007 by David Cournapeu as a Google Summer
of Code project, and since then many volunteers have contributed. See
the AUTHORS file for a complete list of contributors.

It is currently maintained by a team of volunteers.


Download
========

You can download source code and Windows binaries from SourceForge:

http://sourceforge.net/projects/scikit-learn/files/


Dependencies
============

The required dependencies to build the software are python >= 2.5,
NumPy >= 1.1, SciPy and a working C++ compiler.

Optional dependencies are scikits.optimization and the Boost libraries
for module scikits.learn.manifold.

To run the tests you will also need nosetests and python-dap
(http://pypi.python.org/pypi/dap/).


Install
=======

This packages uses distutils, which is the default way of installing
python modules. The install command is::

  python setup.py install


Mailing list
============

There's a general and development mailing list, visit
https://lists.sourceforge.net/lists/listinfo/scikit-learn-general to
subscribe to the mailing list.


IRC channel
===========

Some developers tend to hang around the channel ``#scikit-learn``
at ``irc.freenode.net``, especially during the week preparing a new
release. If nobody is available to answer your questions there don't
hesitate to ask it on the mailing list to reach a wider audience.


Development
===========

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/scikit-learn/scikit-learn.git

or if you have write privileges::

    git clone git@github.com:scikit-learn/scikit-learn.git

Bugs
----

Please submit bugs you might encounter, as well as patches and feature
requests to the tracker located at the address
https://sourceforge.net/apps/trac/scikit-learn/report


Testing
-------

To execute the test suite, run from the project's top directory (you
will need to have nosetest installed)::

    python setup.py test


