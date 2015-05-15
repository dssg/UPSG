==========
User Guide
==========

------
Stages
------

The Stage interface
===================

The Lambda Stage
================

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
    upsg.transform.split.SplitTransTest
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


