===============
Getting Started
===============

------------
Introduction
------------

UPSG (Universal Pipeline for Social Good) is a standard methodology, an
interchange format, and a Python library for writing machine learning 
pipelines. 

It is designed primarily to provide different teams working on different
machine learning problems to share code across different languages
and environments.

------------
Installation
------------

install with::

    pip install git+git://github.com/dssg/UPSG.git

To use the UPSG Python library, we currently require the following packages.
In most environments, pip should take care of this for you.

Required
========

Python packages
---------------
- `Python 2.7 <https://www.python.org/>`_
- `Numpy <http://www.numpy.org/>`_
- `scikit-learn <http://scikit-learn.org/stable/>`_
- `PyTables <https://pytables.github.io/>`_
- `SQLAlchemy <http://www.sqlalchemy.org/>`_

Other packages
--------------
- `HDF5 <https://www.hdfgroup.org/downloads/index.html>`_
 
Optional
========

Python packages
---------------
- `plotlib <http://matplotlib.org/>`_
- `graphviz <https://pypi.python.org/pypi/graphviz>`_
- `pandas <http://pandas.pydata.org/>`_

Other packages
--------------
- `Graphviz <http://www.graphviz.org/>`_

-------
Example
-------

This is how to implement the 
`sklearn "Getting started" pipeline <http://scikit-learn.org/0.10/tutorial.html>`_
