==========
User Guide
==========

------
Stages
------

UPSG programs are a number of pipeline stages that are connected in the
context of a pipeline. Each stage performs one task (for example, reading
data from a csv, imputing data, selecting features, or running an estimator
on data). Pipelines specify which stages should be run in order and with
which data. For example, a simple pipeline might read data from a csv,
impute that data by replacing NaNs with a constant value, and then write 
the imputed data to another csv. 

In UPSG, we would express that as follows::
    
    from upsg.fetch.csv import CSVRead
    from upsg.transform.fill_na import FillNA
    from upsg.export.csv import CSVWrite

    # create csv-reading stage
    stage_read_csv = CSVRead('incomplete_data.csv')

    # create a stage that fill's NaNs with the value 0
    stage_fill_na = FillNA(0)

    # create a stage that writes csvs
    stage_write_csv = CSVWrite('filled_data.csv')

    # create a pipeline
    p = Pipeline()

    # add the stages to the pipeline
    node_read_csv = p.add(stage_read_csv)
    node_fill_na = p.add(stage_fill_na)
    node_write_csv = p.add(stage_write_csv)

    # connect the output of the csv reader to the input of the fill stage
    node_read_csv > node_fill_na

    # connect the output of the fill stage to the input of the csv writer
    node_fill_na > node_write_csv

.. TODO picture of 3-stage
    
After the pipeline is constructed, we run our program and then we can see
our results in filled_data.csv::

    p.run()

Pipelines do not have to be linear, as in the above example. UPSG supports
arbitrary directed acyclic graphs. That means that a pipeline can have
and number of input stages and any number of output stages, and the stages
can be connected in any way as long as there are no loops (cycles).

.. TODO picture of something more complicated


The stage interface
===================

A stage is a program that:

1. Reads zero or more :doc:`.upsg files <file_format>`
2. Writes zero or more :doc:`.upsg files <file_format>`

Each input and each output should have a human-readable lable, or key.
For example, the CSVWrite stage has an output file with the key "output"
and the CSVRead stage has an input file with the key "input." Estimators
have inputs including "X_train", "X_test", "y_train", and "y_test" and
outputs including "y_pred" and "pred_proba". In the Python interface, these
keys can be specified when connecting nodes together. The above example could
have been written::
    
    # connect the output of the csv reader to the input of the fill stage
    node_read_csv['output'] > node_fill_na['input']

    # connect the output of the fill stage to the input of the csv writer
    node_fill_na['output'] > node_write_csv['input']
    
In that example, the .upsg file assigned to node_read_csv's key called 
"output" is selected to be the .upsg file assigned to node_fill_na's key
called "input". Also, the .upsg file assigned to fill_node_na's key called
"output" is selected to be the .upsg file assigned to node_write_csv's key
called "input".

If the keys are omitted, UPSG picks the first key returned by the stage's
.output_keys attribute. Since CSVRead's first output key is "output", 
node_fill_na's first input key is "input", node_fill_na's first output key
is "output", and node_write_csv's first input key is "input", both this code
block and the previous code block have equivalent code.

In principal, any program that fulfills these tasks can be a pipeline Stage.
Programs can be written in bash, R, Python, C, or whatever you like.

As of release 0.0.1, however, stages must be written in Python (or at least
Python wrappers around other :class:`languages <upsg.transform.sql.RunSQL>`). 
The Python class implementing the stage model is the 
:class:`upsg.stage.Stage`.

Python stages should implement either the :class:`upsg.stage.RunnableStage`
interface or the :class:`upsg.stage.MetaStage` interface.

Users can add functionality to their pipelines by either:

1. Selecting a :ref:`pre-existing Stage <_stages_by_module>`. This is a good
   option, provided your venerable devs have anticipated your needs.
2. Implement your own :class:`upsg.stage.Stage`. This option is the most 
   flexible, but also the most involved. See :doc:`implementing_stage`.
3. Wrap arbitrary code inside the :class:`upsg.lambda_stage.LambdaStage`, which
   takes care of some of the boilerplate for you so you don't have to implement
   a full Stage. (see :ref:`here <_lambda_stage>`)

.. _lambda_stage:
The Lambda Stage
================

.. _stages_by_module:
Stages by module
================

:mod:`.export`
--------------

.. autosummary::
    
    upsg.export.csv.CSVWrite
    upsg.export.np.NumpyWrite
    upsg.export.plot.Plot

:mod:`.fetch`
-------------

.. autosummary::

    upsg.fetch.csv.CSVRead
    upsg.fetch.np.NumpyRead
    upsg.fetch.sql.SQLRead

:mod:`.model`
-------------

.. autosummary::
    
    upsg.model.cross_validation
    upsg.model.grid_search
    upsg.model.multiclassify
    upsg.model.multimetric

:mod:`.transform`
-----------------

.. autosummary::

    upsg.transform.apply_to_selected_cols.ApplyToSelectedCols
    upsg.transform.fill_na.FillNA
    upsg.transform.identity.Identity
    upsg.transform.label_encode.LabelEncode
    upsg.transform.lambda_stage.LambdaStage
    upsg.transform.merge.Merge
    upsg.transform.rename_cols.RenameCols
    upsg.transform.split.KFold
    upsg.transform.split.Query
    upsg.transform.split.SplitByInds
    upsg.transform.split.SplitColumns
    upsg.transform.split.SplitTrainTest
    upsg.transform.split.SplitY
    upsg.transform.sql.RunSQL
    upsg.transform.timify.Timify

Wrapping sklearn
================

---------
Pipelines
---------

Connecting stages together
==========================

Running
=======

Visualization and debug output
==============================

-----------------
Universal Objects
-----------------

---------
Utilities
---------


----------------
The Data Toaster
----------------


