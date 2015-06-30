from __future__ import print_function
from collections import namedtuple
import os
import sys
import weakref
import uuid
import abc
import subprocess
import re
import cgi
import itertools as it
import numpy as np

from .uobject import UObjectException
from .utils import get_resource_path

RUN_MODE_ENV_VAR = 'UPSG_RUN_MODE'

class RunMode:
    DBG, LUIGI, LUIGI_QUIET = range(3)
    from_str = {'dbg': DBG, 'luigi': LUIGI, 'luigi_quiet': LUIGI_QUIET}

class PipelineException(Exception):
    pass

class Edge(object):
    """A directed graph edge
    
    Parameters
    ----------
    conn_from : Connection
        The Connection which the edge is directed from
    conn_to : Connection
        The Connection which the edge is directed to
    uid : str or None
        The unique uid of edge. If not provided, a uid will be generated

    Attributes
    ----------
    conn_from : Connection
        The Connection which the edge is directed from
    conn_to : Connection
        The Connection which the edge is directed to
    uid : str or None
        The unique uid of edge. If not provided, a uid will be generated
        
    """

    def __init__(self, conn_from, conn_to, uid=None):
        self.__conn_from = weakref.ref(conn_from)
        self.__conn_to = weakref.ref(conn_to)
        if uid is None:
            uid = 'edge_{}'.format(uuid.uuid4())
        self.__uid = uid

    @property
    def conn_from(self):
        return self.__conn_from

    @property
    def conn_to(self):
        return self.__conn_to

    @property
    def uid(self):
        return self.__uid


class Connection(object):

    """Object signifying a connection between two Nodes in a pipeline.

    A Connection is not a graph edge. It's more accurate to say that a
    Connection is one side of an edge. For example, say we have Nodes A
    and B. Node A has an output key 'output' and Node B has an input key 
    'input'. Then, node A has a Connection A['output'] and Node B has a 
    Connection B['input'].  In order to create an edge between Node A and 
    Node B, we do:

    >>> A['output'].connect_to(B['input'])

    or, alternatively,

    >>> A['output'] > B['input']
    
    Parameters
    ----------
    key : str
        The name of the input key or output key that that this Connection
        signifies
    outgoing : bool
        True if the edge is outgoing, False if the edge is incoming
    node : Node
        The Node that owns this connection

    Attributes
    ----------
    outgoing : bool
        True if the edge is outgoing, False if the edge is incoming
    other : Connection or None
        The connection to which this Connection connects
    key : str
        This connection's key
    node : Node
        The Node that owns this connection
    edge : Edge
        The Edge between this connection and other

    """

    def __init__(self, key, outgoing, node):
        self.__key = key
        self.__other = None
        self.__outgoing = outgoing
        self.__node = weakref.ref(node)
        self.__edge = None

    def connect_to(self, other):
        """
        
        If other is a Connection, create an edge between self.node and 
        other.node using a link between the Connections self and other.

        If other is a Node, create an edge between self and other[key],
        where key is the first input key of other.

        Parameters
        ----------
        other : Connection or Node

        """
        if isinstance(other, Node):
            other = other[other.input_keys[0]]

        if not self.outgoing:
            raise PipelineException("Can't connect from an incoming "
                                    "edge")
        if other.outgoing:
            raise PipelineException("Can't connect to an outgoing "
                                    "edge")
        self.__other = other
        other.__other = self
        edge = Edge(self, other)
        self.__edge = edge
        other.__edge = edge

    def __gt__(self, other):
        """Synonym for self.connect_to(other)"""
        self.connect_to(other)

    @property
    def outgoing(self):
        return self.__outgoing

    @property
    def other(self):
        return self.__other

    @property
    def key(self):
        return self.__key

    @property
    def node(self):
        return self.__node()

    @property
    def edge(self):
        return self.__edge


class Node(object):

    """A real or virtual Node the Pipeline graph

    Parameters
    ----------
    stage : Stage
        The Stage to be run when this Node is executed
    connections : dict of (str : Connection) or None
        If provided, the Node will use the provided connections rather than
        making its own. This is useful for virtual nodes which manage
        Connections for a subgraph of the Pipeline rather than for an
        individual stage.
    label : str or None
        If provided, will be returned by this node's __str__ method. 
        Otherwise, will default to using __repr__
    uid : str or None
        The unique uid of edge. If not provided, a uid will be generated

    """


    def __init__(self, stage, connections=None, label=None, uid=None):
        self.__stage = stage
        self.__connections = {}
        if connections is None:
            self.__connections.update({key: Connection(key, False, self)
                                       for key in stage.input_keys})
            self.__connections.update({key: Connection(key, True, self)
                                       for key in stage.output_keys})
        else:
            self.__connections.update(connections)
        self.__label = label
        if uid == None:
            uid = 'node_{}'.format(uuid.uuid4())
        self.__uid = uid    

    def __getitem__(self, key):
        """Gets the Connections specified by key"""
        return self.__connections[key]

    def __call__(self, *args, **kwargs):
        """Alternative syntax for connecting Stages together

        Parameters
        ----------
        args : list of (Node or Connection)
            The nth output_key of self.output_keys will be connected to the
            nth node or connection. For example, if 
            
            >>> self.output_keys == ['output', 'complement', 'status']
            
            and we invoke 

            >>> self(clf_node['X_train'], clf_node['X_test'], status_node),

            it is equivalent to doing:

            >>> self['output'] > clf_node['X_train'] 
            >>> self['complement'] > clf_node['X_test']
            >>> self['status'] > status_node

        kwargs : dict of str: (Node or Connection)
            The output key corresponding to the given keyword will be 
            connected to the argument assigned to that output key. For
            example, if we invoke 
            
            >>> self(complement=clf_node['X_test'], 
            ...     status=status_node, output=clf_node['X_train']) 

            it is equivalent to doing:

            >>> self['complement'] > clf_node['X_test']
            >>> self['status'] > status_node
            >>> self['output'] > clf_node['X_train'] 

        """
        input_keys = self.input_keys
        output_keys = self.output_keys
        for i, arg in enumerate(args):
            arg > self[input_keys[i]]
        for key in kwargs:
             kwargs[key] > self[key]
        return self

    def __repr__(self):
        return 'Node({})'.format(self.__stage)

    def __str__(self):
        if self.__label is None:
            return self.__repr__()
        return self.__label

    def get_stage(self):
        return self.__stage

    def get_inputs(self, live_only=True):
        """Returns a dict of (key : Connection) for all Connections
        that are incoming.

        Parameters
        ----------
        live_only: bool
            if True, will only return Connections that are actually connected
                to another node.

        """
        # TODO raise an error if all of the required inputs have not been
        # connected yet
        return {
            key: self.__connections[key] for key in self.__connections if (
                not self.__connections[key].outgoing) and (
                (not live_only) or (
                    self.__connections[key].other is not None))}

    def get_outputs(self, live_only=True):
        """

        Parameters
        ----------
        live_only: bool
            if True, will only return Connections that are actually connected
                to another node.

        Returns
        -------
        dict of (key: Connection)
            represents outgoing connections

        """
        return {key: self.__connections[key] for key in self.__connections
                if self.__connections[key].outgoing and
                (not live_only or self.__connections[key].other is not None)}

    @property
    def output_keys(self):
        return self.get_stage().output_keys

    @property
    def input_keys(self):
        return self.get_stage().input_keys

    @property
    def uid(self):
        return self.__uid

    def connect_to(self, other):
        """Invokes the connect_to method for the first outgoing connection
        of this node

        Parameters
        ----------
        other : Connection or node

        """
        self[self.output_keys[0]].connect_to(other)

    def __gt__(self, other):
        """Synonym for self.connect_to(other)"""
        self.connect_to(other)

class Pipeline(object):

    """Internal representation of a UPSG pipeline.

    Our structure is merely a graph of pipeline elements. Execution will
    be relegated to either an external graph processing framework 
    or some simplified, internal replacement.

    """

    def __init__(self):
        self.__nodes = []

    def __struct_str_rep(self, pipeline):
        return {str(node) : {key: (str(conn.other.node), conn.other.key) for
                             key, conn in node.get_outputs().items()}
                for node in pipeline.__nodes}

    def is_equal_by_str(self, other):
        """
        
        Returns whether or not self has the same nodes and edges as 
        Pipeline other. Node equality is determined by whether nodes have the
        same str representation, so it's really only useful in contrived 
        circumstances like our unit tests
        
        """
        return self.__struct_str_rep(self) == self.__struct_str_rep(other)
    def add(self, stage, label=None):
        """Add a stage to the pipeline

        Parameters
        ----------
        stage: Stage
            Stage to add to the pipeline
        label: str or None
            label to be returned by created Node's __str__ method. If not
            provieded, will use Node's __repr__ method

        Returns
        -------
        Node 
            A Node encapsulating the given stage

        """
        # TODO this is here to avoid a circular import. Should refactor
        from stage import MetaStage, RunnableStage
        if isinstance(stage, RunnableStage):
            node = Node(stage, label=label)
            self.__nodes.append(node)
            return node
        if isinstance(stage, MetaStage):
            metanode = self.__integrate(stage, *stage.pipeline)
            return metanode
        raise TypeError('Not a valid RunnableStage or MetaStage')

    def __integrate(self, stage, other, in_node, out_node):
        """Integrates another pipeline into this one and creates a virtual
        uid to access the sub-pipeline.

        Parameters
        ----------
        stage : MetaStage 
        other : Pipeline
        in_node : Node
            node responsible for delivering input to the Pipeline
        out_node : Node
            node responsible for collection output from the pipline

        Returns
        -------
        Node 
            A nodewhich can be used to connect nodes to sub-pipeline as if
            the sub-pipeline were a single node.

        """
        self.__nodes += other.__nodes
        connections = {}
        connections.update(in_node.get_inputs(False))
        connections.update(out_node.get_outputs(False))
        return Node(stage, connections=connections)

    def visualize(self, filename=None, html_map=False):
        """Creates a pdf to vizualize the pipeline.

        Requires the graphviz python package: 
        (https://pypi.python.org/pypi/graphviz) 
        and Graphviz:
        (http://www.graphviz.org/)


        parameters
        ----------
        filename : str or None
            File name for the rendered pdf. If not given, a file name will be
            selected automatically

        Returns
        -------
        str
            potentially relative path of the rendered pdf

        """
        from graphviz import Digraph
        from os import system
        dot = Digraph(name='G')
        node_names = {}
        next_node_number = 0
        node_queue = [node for node in self.__nodes
                      if not node.get_outputs()]  # start with the root nodes
        for node in node_queue:
            name = 'node_{}'.format(next_node_number)
            node_names[node] = name
            dot.node(name, str(node), shape='box', URL='#{}'.format(node.uid))
            next_node_number += 1
        processed = set()
        while node_queue:
            node = node_queue.pop()
            if node in processed:
                continue
            name = node_names[node]
            input_connections = node.get_inputs()
            input_nodes = frozenset([input_connections[input_key].other.node
                                     for input_key in input_connections])
            for input_key in input_connections:
                conn = input_connections[input_key]
                other_conn = conn.other
                other_node = other_conn.node
                try:
                    other_name = node_names[other_node]
                except KeyError:
                    other_name = 'node_{}'.format(next_node_number)
                    next_node_number += 1
                    dot.node(other_name, str(other_node), shape='box',
                             URL='#{}'.format(other_node.uid))
                    node_names[other_node] = other_name
                dot.edge(other_name, name, label='{}\n::\n{}'.format(
                    other_conn.key, 
                    conn.key), URL='#{}'.format(conn.edge.uid))
            node_queue.extend(input_nodes)
            processed.add(node)
        if html_map:
            dot_file = dot.save(filename)
            map_file = dot_file + '.html'
            png_file = dot_file + '.png'
            scale = 1.0
            # Some help from:
            # http://stackoverflow.com/questions/15837283/graphviz-embedded-url
            proc = subprocess.Popen(
                    ['dot', '-Tpng', dot_file, '-o', png_file],
                    stderr=subprocess.PIPE)
            _, stderr = proc.communicate()
            if 'Scaling' in stderr: 
                # dot has decided to scale the image, so we'll have to
                # scale the map
                scale = float(re.search('\d+(\.\d*)?', stderr).group(0))
            proc = subprocess.call(
                    ['dot', '-Tcmapx', dot_file, '-o', map_file])
            return (png_file, map_file, scale)    
        out_file = dot.render(filename = filename)
        return out_file

    def run_debug(self, **kwargs):
        """Run the pipeline in the current Python process.

        This method of running the job runs everything in serial on a single
        process. It is provided for debugging purposes for use with small jobs.
        For larger and more performant jobs, use the run method.

        Parameters
        ----------
        output : str
            Method of displaying output. One of:

                'bw'
                    prints progress and truncated stage output to terminal
                'color'
                    prints progress and truncated stage output 
                    to terminal using ANSI colors
                'progress'
                    only prints progress
                'html'
                    prints pipeline visualization and truncated output
                    in an html report. Also prints progress to terminal
                'silent'
                    prints no output.
                '' [empty string] 
                    If the environmental variable UPSG_DEBUG_OUTPUT_MODE is set
                    to one of the above strings, then the value of 
                    UPSG_DEBUG_OUTPUT_MODE will determine the output method.
                    If the environmental variable is not set or is invalid, then
                    the effect will be the same as specifying 'silent'
        
        report_path : str
            If output='html', the path of the html file to be generated.
            If unspecified, will use graph_out.html in the current working
            directory

        single_step : bool
            If True, will invoke pdb after every stage is run

        """
        import run_debug
        run_debug.run(self, self.__nodes, **kwargs)

    def run_luigi(self, **kwargs):
        """Run pipeline using luigi

        Parameters
        ----------
        logging_conf_file : str or None
            Path of file to configure luigi logging that follows the format 
            of: https://docs.python.org/2/library/logging.config.html
        worker_processes : int or None
            Number of processes to run on. If None, will pick the number
            of cpus on the host system
        """

        import run_luigi
        run_luigi.run(self.__nodes, **kwargs)

    def run_luigi_quiet(self, **kwargs):
        """Run a pipeline using luigi using a default logging configuration.
        
        The logging configuration notes events of level ERROR or higher
        in the file upsg_luigi.log in the current working directory

        Parameters
        ----------
        worker_processes : int or None
            Number of processes to run on. If None, will pick the number
            of cpus on the host system
        """

        kwargs['logging_conf_file'] = get_resource_path(
                'luigi_default_logging.cfg')
        self.run_luigi(**kwargs)

    RUN_METHODS = {RunMode.DBG: run_debug,
                   RunMode.LUIGI: run_luigi,
                   RunMode.LUIGI_QUIET: run_luigi_quiet}

    def run(self, run_mode=None, **kwargs):
        """Run the pipeline
        
        Parameters
        ----------
        run_mode : {RunMode.DBG, RunMode.LUIGI} or str or None
            Specifies the method to use to run the pipeline. 
            If an attribute of RunMode, specifies the run mode to use.
            If a str, should be either 'dbg' or 'luigi'
            If None, defaults to debug unless the environmental variable:
            UPSG_RUN_MODE is set, which should be either 'dbg' or 'luigi' and
            will specify the run mode
        kwargs : kwargs
            keyword arguments to pass to the run method

        """
        if run_mode is None:
            try: 
                run_mode = os.environ[RUN_MODE_ENV_VAR]
            except KeyError:
                run_mode = 'dbg'

        if isinstance(run_mode, basestring):
            try:
                run_mode = RunMode.from_str[run_mode.lower()]
            except KeyError:
                run_mode = RunMode.DBG

        self.RUN_METHODS[run_mode](self, **kwargs)
