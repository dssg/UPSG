============================================
UPSG: The Universal Pipeline for Social Good
============================================

------------
Introduction
------------

UPSG is a standard methodology, an interchange format, and a Python library for
writing machine learning pipelines. 

It is designed primarily to provide different teams working on different
machine learning problems a way to share code across different languages
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
`sklearn "Getting started" pipeline <http://scikit-learn.org/0.10/tutorial.html>`_::

    digits = datasets.load_digits()
    digits_data = digits.data
    # for now, we need a column vector rather than an array
    digits_target = digits.target

    p = Pipeline()

    # load data from a numpy dataset
    stage_data = NumpyRead(digits_data)
    stage_target = NumpyRead(digits_target)

    # train/test split
    stage_split_data = SplitTrainTest(2, test_size=1, random_state=0)

    # build a classifier
    stage_clf = wrap_and_make_instance(SVC, gamma=0.001, C=100.)

    # output to a csv
    stage_csv = CSVWrite('out.csv')

    node_data, node_target, node_split, node_clf, node_csv = map(
        p.add, [
            stage_data, stage_target, stage_split_data, stage_clf,
            stage_csv])

    # connect the pipeline stages together
    node_data['output'] > node_split['input0']
    node_target['output'] > node_split['input1']
    node_split['train0'] > node_clf['X_train']
    node_split['train1'] > node_clf['y_train']
    node_split['test0'] > node_clf['X_test']
    node_clf['y_pred'] > node_csv['input']

    p.run()
    
    # results are now in out.csv

----------
Next Steps
----------

Check out the `documentation <http://dssg.io/UPSG>`_
