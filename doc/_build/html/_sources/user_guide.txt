==========
User Guide
==========

.. _stages:

------
Stages
------

UPSG programs are a number of pipeline stages that are connected in the context
of a pipeline. Each stage performs one task (for example, reading data from a
csv, imputing data, selecting features, or running an estimator on data).
Pipelines specify which stages should be run, in order they should be run, and
with which data they should be run. For example, a simple pipeline might read
data from a csv, impute that data by replacing NaNs with a constant value, and
then write the imputed data to another csv. 

In UPSG, we would express that as follows::

    from upsg.pipeline import Pipeline
    
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

.. figure:: images/three_stage.png
   :align: center
    
   A simple pipeline which reads from a csv, fills NaN entries, and then
   writes to a different csv.

After the pipeline is constructed, we run our program and then we can see
our results in filled_data.csv::

    p.run()

Pipelines do not have to be linear, as in the above example. UPSG supports
arbitrary directed acyclic graphs. That means that a pipeline can have
any number of input stages and any number of output stages, and the stages
can be connected in any way as long as there are no loops (cycles).

.. figure:: images/complicated.png
   :height: 500px
   :align: center

   A more complicated pipeline

The stage interface
===================

A stage is a program that:

1. Reads zero or more :doc:`.upsg files <file_format>`
2. Writes zero or more .upsg files

Each input and each output should have a human-readable label, or key.
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
The Python classes implementing the stage model are
:class:`upsg.stage.MetaStage` and :class:`upsg.stage.RunnableStage`.

Python stages should be subclasses of either :class:`upsg.stage.RunnableStage`
or :class:`upsg.stage.MetaStage`.

Users can add functionality to their pipelines by either:

1. Selecting a :ref:`pre-existing Stage <stages_by_module>`. This is a good
   option, provided your venerable devs have anticipated your needs.
2. :ref:`Wrapping <wrapping_sklearn>` an sklearn estimator or metric
3. Implement your own Stage. This option is the most 
   flexible, but also the most involved. See :doc:`implementing_stage`.
4. Wrap arbitrary code inside the 
   :class:`upsg.transform.lambda_stage.LambdaStage`, which
   takes care of some of the boilerplate for you so you don't have to implement
   a full Stage. (see :ref:`here <lambda_stage>`)

.. _lambda_stage:

The LambdaStage
================

The :class:`LambdaStage <upsg.transform.lambda_stage.LambdaStage>` class
provides a way to wrap arbitrary code in the UPSG framework with minimal 
boilerplate. In order to initialize a LambdaStage, the user must provide:

1. A Function that takes zero or more numpy 
   `structured arrays <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_
   and returns either:

    1. A numpy array, or,
    2. A tuple of numpy arrays.

2. Either a list of output keys or the number of outputs to expect

Using LambdaStage, any function that takes and returns Numpy arrays can be
seamlessly incorporated into UPSG. See the API for more details.

.. _stages_by_module:

Stages by module
================

A number of other stages that perform common tasks have already been
implemented. They are listed below.

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
    
    upsg.model.cross_validation.CrossValidationScore
    upsg.model.grid_search.GridSearch
    upsg.model.multiclassify.Multiclassify
    upsg.model.multimetric.Multimetric

:mod:`.transform`
-----------------

.. autosummary::

    upsg.transform.apply_to_selected_cols.ApplyToSelectedCols
    upsg.transform.fill_na.FillNA
    upsg.transform.generate_feature.GenerateFeature
    upsg.transform.hstack.HStack
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

.. _wrapping_sklearn:

Wrapping sklearn
================

By using the :func:`upsg.wrap.wrap_sklearn.wrap` function or the 
:func:`upsg.wrap.wrap_sklearn.wrap_and_make_instance` function we can make
Stages from sklearn estimators, metrics, or partition iterators with only a few 
lines of code. See API for more details

---------
Pipelines
---------

:class:`Pipelines <upsg.pipeline.Pipeline>` are the way that stages are
organized into workflow. UPSG programs usually have six phases:

1. Setup

   >>> from upsg.pipeline import Pipeline
   >>> from upsg.fetch.csv import CSVRead
   >>> from upsg.export.csv import CSVWrite

2. Create a Pipeline

   >>> p = Pipeline()

3. Create all the Stages

   >>> stage_read = CSVRead('in.csv')
   >>> stage_write = CSVWrite('out.csv')

4. Add Stages to a Pipeline, creating :class:`Nodes <upsg.pipeline.Node>`

   >>> node_read = p.add(stage_read)
   >>> node_write = p.add(stage_write)

5. Connect nodes

   >>> node_read > node_write

6. Run the pipeline

   >>> p.run()

Phase 2 merely initializes a Pipeline.

Phase 3 creates a number of stages, as discussed in :ref:`stages`.

Phase 4 adds Stages to the pipeline using the 
:func:`upsg.pipeline.Pipeline.add` method. Each stage must be added to a 
Pipeline once. The return value of Pipeline.add will be a Node, which is used
to connect pipeline stages together.

Phase 5 connects nodes together. It is discussed in more detail 
:ref:`below <connecting_stages_together>`.

Phase 6 invokes :func:`upsg.pipeline.Pipeline.run`. This is discussed in more
detail :ref:`below <running>`.

.. _connecting_stages_together:

Connecting stages together
==========================

Once we have added our Stages to the pipeline and collected a number of 
:class:`Nodes <upsg.pipeline.Node>`, we can connect our Nodes together in
order to specify the dependencies between Stages. For example, if we have 
a Stage :code:`stage_a` and a Stage :code:`stage_b`, where :code:`stage_a` has 
an output named "out_a" and :code:`stage_b` has an input named "in_b_1", and 
:code:`stage_b` expects that it's input "in_b_1" will be provided by 
:code:`stage_a`'s "out_a" output, then we can connect the two like:

>>> from upsg.pipeline import Pipeline
>>> p = Pipeline()
>>> node_a = p.add(stage_a)
>>> node_b = p.add(stage_b)
>>> node_a['out_a'] > node_b['in_b_1']

.. figure:: images/connecting1.png
   :align: center

   stage_a's output "out_a" is connected to stage_b's input "in_b_1"

Further, if :code:`stage_b` also takes an input called "in_b_2", which is
supposed to be provided by the "out_c" argument of :code:`stage_c`, we can
connect it like this:

>>> node_c = p.add(stage_c)
>>> node_c['out_c'] > node_b['in_b_2']

.. figure:: images/connecting2.png
   :align: center

   In addition, stage_c's output "out_c" is connected to stage_b's input
   "in_b_2"

For convenience, there are a few alternative syntaxes to express the same thing
expressed above.

If you intend to use the first key returned by :code:`stage.input_keys` or 
:code:`stage.output_keys` for some stage, then the key of the input argument
or the output argument can be omitted. For example, let's assume the following
input and output keys:

>>> stage_a.output_keys
['out_a']
>>> stage_c.output_keys
['out_c']
>>> stage_b.input_keys
['in_b_1', 'in_b_2']

Then, we could do the same thing as we did above by writing:

>>> node_a > node_b
>>> node_c > node_b['in_b_2']

Note that "in_b_2" still needs to be explicitly specified, since it is not the
first key in stage_b.input_keys

We also support function notation. For example:

>>> node_a > node_b

is the same as

>>> node_b(node_a)

and 

>>> node_a['out_a'] > node_b['in_b_1']

is the same as

>>> node_b(in_b_1=node_a['out_a'])

and

>>> node_a > node_b['in_b_1']
>>> node_c > node_b['in_b_2']

is the same as

>>> node_b(in_b_1=node_a, in_b_2=node_b)

.. _running:

Running
=======

UPSG is designed to allow for a number of ways to run pipelines. For
example, a pipeline may be run in a shared-memory system using unix pipes, or
it may be run on a cluster by scheduling a number of Hadoop jobs. 

The :class:`upsg.pipeline.Pipeline` class will provide one method for each of
these ways to run the pipeline. The method :func:`upsg.pipeline.Pipeline.run`
will always provide a default run method that is functionally correct.

As of version 0.0.1, There are two run methods implemented: 
:func:`upsg.pipeline.Pipeline.run_debug` and 
:func:`upsg.pipeline.Pipeline.run_luigi`. The run_debug method runs the 
pipeline on one core in serial, and is not suitable for extremely large jobs,
but it does provide a number of tools to ensure that a pipeline is running
correctly. These are discussed in more detail 
:ref:`below <visualizing_and_debug>`. The run_luigi method utilizes the luigi
package run tasks in a scalable way.

Environmental Variables
-----------------------

UPSG provides environmental variables to determine default behavior of the 
:func:`upsg.pipeline.Pipeline.run` method. The variables are:

**UPSG_RUN_MODE**
    Determines the method to use to run the pipeline. Should be either:

    *dbg* 
        for the debug method
    *luigi* 
        for running using luigi

**UPSG_DEBUG_OUTPUT_MODE**
    If the pipeline is run in debug mode, and the ``output`` argument of
    :func:`upsg.pipeline.Pipeline.run_debug` is not specified, this variable
    will determine the argument of ``output``. Should be one of:

    *bw*
        prints progress and truncated stage output to terminal
    *color*
        prints progress and truncated stage output 
        to terminal using ANSI colors
    *progress*
        only prints progress
    *html*
        prints pipeline visualization and truncated output
        in an html report. Also prints progress to terminal
    *silent*
        prints no output.


.. _visualizing_and_debug:

Visualization and debug output
==============================

There are several tools provided to help ensure that a pipeline you have built
is working correctly. 

One is the :func:`upsg.pipeline.Pipeline.visualize` method, which will use
Graphviz to create a graph visualization of the Pipeline. Each node on the
output graph is a Stage, and each edge is a passed .upsg file. The labels of
the edge are in the format::

    name of the output stage's output key
    ::
    name of the input stage's input key

.. figure:: images/visualize.png
   :align: center

   In the above example, there is a Stage called "read_in" and a stage called 
   "write_out". "read_in" has an output argument called "output" which is 
   connected to the input argument "input" of "write_out".

One thing that might be useful to make graphs is to utilize the optional second
argument of :func:`upsg.pipeline.Pipeline.add`. The second argument of 
Pipeline.add allows the user to provide a label which will be used as the name
of the node.

The other debugging tool is the various types of output that can be provided 
by :func:`upsg.pipeline.Pipeline.run_debug`. Set the "output" argument to
one of the supported strings to get some feedback on what the pipeline is
doing.

.. TODO explain the html report

-----------------
Universal Objects
-----------------

The primary way that the UPSG Python library interfaces with .upsg files is
through the :class:`Universal Object <upsg.uobject.UObject>` or UObject.
Conceptually, the UObject is a write-once variable that is backed by a .upsg
file. The UObject can be written to using one of its "from\_" methods, and then
read from using one of its "to\_" methods. You will generally not have to deal
with UObjects unless you :doc:`implement your own Stage <implementing_stage>`.

---------
Utilities
---------

There are a number of exposed utility functions which may be useful, especially
if you are :doc:`implement your own Stage <implementing_stage>`. Most of them
involve manipulating the types of Numpy 
`structured arrays <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_.
These are in the :mod:`upsg.utils` module.

.. autosummary::

    upsg.utils.datetime64_to_datetime
    upsg.utils.dict_to_np_sa
    upsg.utils.get_resource_path
    upsg.utils.html_escape
    upsg.utils.import_object_by_name
    upsg.utils.is_sa
    upsg.utils.np_dtype_is_homogeneous
    upsg.utils.np_nd_to_sa
    upsg.utils.np_sa_to_dict
    upsg.utils.np_sa_to_nd
    upsg.utils.np_to_sql
    upsg.utils.np_type
    upsg.utils.obj_to_str
    upsg.utils.random_table_name
    upsg.utils.sql_to_np
    upsg.utils.utf_to_ascii    

----------------
The Data Toaster
----------------

The :class:`upsg.toaster.DataToaster` is an interface designed to build 
pipelines without doing the process explicitly. Rather than interacting with
Stages, Nodes, and Connections, the user interacts with a DataToaster, which
resembles a 
`Pandas DataFrame <http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html>`_.
As of release 0.0.1, functionality is limited, but it can be used for some
simple tasks::

        from upsg.toaster import DataToaster
        dt = DataToaster()
        # Read in a csv
        dt.from_csv('test_toaster.csv')
        # Training is data before 2006-06-15; testing is after. The column
        # giving us classification is 'cat'
        dt.split_by_query('cat', "date < DT('2006-06-15')")
        # Select features (manually, in this case)
        dt.transform_select_cols(('factor_1', 'factor_2'))
        # Do some last-minute cleanup
        dt.transform_with_sklearn('sklearn.preprocessing.StandardScaler')
        # Try a bunch of classifiers and parameters. Output to report.html
        dt.classify_and_report(report_file_name='report.html')
        dt.run()
