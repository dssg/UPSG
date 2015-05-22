====================
Implementing a Stage
====================

In order to implement a Stage, you should subclass either:

1. :class:`upsg.stage.RunnableStage`, or
2. :class:`upsg.stage.MetaStage`. 

In either class, a sublass must implement two properties:

1. .input_keys, and
2. .output_keys

The former returns a list of the names of the input arguments that the class
will take; the latter returns a list of the output arguments that the class
will provide. An important note is that all of the input arguments requested
here will not necessarily be provided by the pipeline, and all output arguments
will not necessarily be sent elsewhere. 

All input keys and output keys must be unique. There cannot be two input keys
with the same name, two output keys with the same name, or an input key
with the same name as an output key.

For example, say we want to write a Stage that takes two structured arrays,  
performs an element-wise sum of the two arrays, and an element-wise multiply of 
the two arrays. We have two input arguments named "input0" and "input1", and
two output arguments named "sum" and "product". A partial implementation would
be::

    from upsg.stage import RunnableStage

    class SumAndMult(RunnableStage):
        @property
        def input_keys(self):
            return ['input0', 'input1']

        @property
        def output_keys(self):
            return ['sum', 'product']

        ...

The rest of implementation depends on whether we are implementing a 
RunnableStage or a MetaStage

----------------------------
Implementing a RunnableStage
----------------------------

To implement a RunnableStage, the class must, in addition to input_keys and
output_keys, implement the :func:`upsg.stage.RunnableStage.run` method, which
takes:

1. outputs_requested, A list of requested output keys, and
2. a dict of (input keys: :class:`UObjects <upsg.uobject.UObject>`) passed as
   keyword arguments.

The method is expected to return a dict of (output keys: UObjects).

The outputs_requested argument informs the Stage which of its output keys are
connected elsewhere in the pipeline. If it takes a significant amount of 
computation to calculate one of the outputs and it is not present in 
outputs_requested, then the computation can be omitted and the output argument
need not be returned.

For each of the input keys, if it has been provided by the pipeline, 
:code:`kwargs[input_key]` should return a UObject corresponding to that input
key. That UObject can then be interpreted using one of the UObject's to\_
methods. For example, in order to interpret the input argument corresponding
to input0 as a numpy array, we could do::

    array_0 = kwargs['input0'].to_np()

In order to interpret input1 as an sql table, we could do::

    sql_1 = kwargs['input1'].to_sql()

For each output argument, the run method is expected to initialize a UObject 
(in the :class:`UObjectPhase.Write <upsg.uobject.UObjectPhase>` phase), store
some data in the UObject with one of its from\_ methods, and then put that
UObject in the returned dictionary with its output key as a key. For example,
if we wanted to return the variable :code:`calc_sum`, which was encoded as a 
Numpy array, using the output key "sum" we would do::

    uo_sum = UObject(UObjectPhase.Write)
    uo_sum.from_np(calc_sum)
    return {'sum': uo_sum}

A full implementation of the example we started in the previous section
(assuming we have defined :code:`elmtwise_sum` and :code:`elmtwise_prod` 
somewhere else) would be::

    from upsg.stage import RunnableStage
    from upsg.uobject import UObject, UObjectPhase

    class SumAndMult(RunnableStage):
        @property
        def input_keys(self):
            return ['input0', 'input1']

        @property
        def output_keys(self):
            return ['sum', 'product']

        def run(self, outputs_requested, **kwargs):
            array_1 = kwargs['input0'].to_np()
            array_2 = kwargs['input1'].to_np()
            to_return = {}
            # only calculate the sum if somebody requests it
            if 'sum' in outputs_requested:
                uo_sum = UObject(UObjectPhase.Write)
                uo_sum.from_np(elmtwise_sum(array_1, array_2))
                to_return['sum'] = uo_sum
            # only calculate the product if somebody requests it
            if 'product' in outputs_requested:
                uo_prod = UObject(UObjectPhase.Write)
                uo_prod.from_np(elmtwise_prod(array_1, array_2))
                to_return['product'] = uo_prod
            return to_return

------------------------
Implementing a MetaStage
------------------------

MetaStages do not implement the run method. Rather, they build their own,
inner Pipelines, which will be transparently embedded in a larger, outer
Pipeline when the MetaStage is added to the outer Pipeline. MetaStages must 
implement the pipeline property, which returns a tuple::
    
    (inner_pipeline, entry_node, exit_node)

The inner Pipeline will be embedded in some outer Pipeline when the outer 
Pipeline's .add method is called.

To the outer Pipeline, the MetaStage will look like a single Stage rather than
being the collection of Stages that it actually is. Consequently, the MetaStage
must select an entry_node and an exit_node that will collectively appear to the
outer Pipeline as a single node. All the input sent by the outer Pipeline to
the MetaStage will be delivered to the entry_node as input arguments. All the
output generated by the MetaStage will come from the exit_node as output 
arguments.

For example, we will implement the previous example as a MetaStage rather than
a RunnableStage using :class:`upsg.transform.identity.Identity` and 
:class:`upsg.transform.lambda_stage.LambdaStage`. In this example, the
element-wise multiplication and the element-wise addition are performed in 
separate, parallel Stages, allowing for the two operations to be performed in
parallel if the scheduler chooses to do so::

    from upsg.pipeline import Pipeline
    from upsg.stage import MetaStage
    from upsg.uobject import UObject, UObjectPhase
    from upsg.transform.identity import Identity
    from upsg.transform.lambda_stage import LambdaStage

    class SumAndMult(MetaStage):
        @property
        def input_keys(self):
            return ['input0', 'input1']

        @property
        def output_keys(self):
            return ['sum', 'product']

        @property
        def pipeline(self):
            return (self.__pipeline, self.__entry_node, self.__exit_node) 

        def __init__(self):
            # Our entry stage passes input arguments to the rest of the
            # pipeline without altering them
            entry_stage = Identity(input_keys=['input0', 'input1'])

            # Our LambdaStages take input from the entry stage and then
            # do the actual work
            sum_stage = LambdaStage(
                lambda input0, input1: elmtwise_sum(input0, input1),
                ['sum'])
            prod_stage = LambdaStage(
                lambda input0, input1: elmtwise_prod(input0, input1),
                ['product'])

            # Our exit stage collects output arguments from our Lambda
            # stages and then passes them to the outer pipeline with the
            # correct name
            exit_stage = Identity(output_keys=['sum', 'product'])

            # Initialize the inner pipeline
            self.__pipeline = Pipeline()

            # Add all our stages
            self.__entry_node = self.__pipeline.add(entry_stage)
            sum_node = self.__pipeline.add(sum_stage)
            prod_node = self.__pipeline.add(prod_stage)
            self.__exit_node = self.__pipeline.add(exit_stage)

            # connect our entry node to our LambdaStage nodes
            self.__entry_node['input0_out'] > sum_node['input0']
            self.__entry_node['input1_out'] > sum_node['input1']
            self.__entry_node['input0_out'] > prod_node['input0']
            self.__entry_node['input1_out'] > prod_node['input1']

            # connect our LambdaStage nodes to our exit node
            sum_node['sum'] > self.__exit_node['sum_in']
            prod_node['product'] > self.__exit_node['product_in']

We can then connect our MetaStage to an outer pipeline as if it were a single 
node::

    from upsg.fetch.csv import CSVRead
    from upsg.export.csv import CSVWrite
    
    # Initialize the outer pipeline
    p = Pipeline()

    # Read our inputs from csv
    read_input0_from_csv = p.add(CSVRead('input0.csv'))
    read_input1_from_csv = p.add(CSVRead('input1.csv'))

    # Initialize and add our SumAndMult Stage
    sum_and_mult = p.add(SumAndMult())

    # Write our results to csvs
    write_sum_to_csv = p.add(CSVWrite('sum.csv'))
    write_prod_to_csv = p.add(CSVWrite('prod.csv'))

    # Connect everything up. Notice that sum_and_mult looks like a 
    # single node, even though it is actually an entire Pipeline
    read_input0_from_csv > sum_and_mult['input0']
    read_input1_from_csv > sum_and_mult['input1']
    sum_and_mult['sum'] > write_sum_to_csv
    sum_and_mult['product'] > write_prod_to_csv

We end up with a pipeline that looks like this:

.. figure:: images/metastage.png
   :align: center

   The flattened outer pipeline after adding our MetaStage. Nodes that come
   from the MetaStage are highlighted in yellow.

The Pipeline that we constructed in our MetaStage has been transparently 
embedded in our outer pipeline. Using MetaStages, it is possible to 
automatically populate a Pipeline with thousands of Stages without explicit
user intervention.
