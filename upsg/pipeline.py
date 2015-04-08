from __future__ import print_function
import os
import pdb
from collections import namedtuple
import itertools as it
import weakref
import numpy as np


class PipelineException(Exception):
    pass


class Connection:

    """Object signifying a connection between two Nodes in a pipeline.

    A Connection is not a graph edge. It's more accurate to say that a
    Connection is one side of an edge. For example, say we have Nodes A
    and B. Node A has an output key 'out' and Node B has an input key 'in'.
    Then, node A has a Connection A['out'] and Node B has a Connection B['in'].
    In order to create an edge between Node A and Node B, we do:

    A['out'].connect_to(B['in'])

    or, alternatively,

    A['out'] > B['in']

    """

    def __init__(self, key, outgoing, node):
        """

        Parameters
        ----------
        key: str
            The name of the input key or output key that that this Connection
            signifies
        outgoing: bool
            True if the edge is outgoing, False if the edge is incoming
        node: Node
            The Node that owns this connection

        """
        self.__key = key
        self.__other = None
        self.__outgoing = outgoing
        self.__node = weakref.ref(node)

    def connect_to(self, other):
        """Create an edge between self.node and other.node using a link
        between the Connections self and other

        Parameters
        ----------
        other : Connection

        """
        if not self.outgoing:
            raise PipelineException("Can't connect from an incoming "
                                    "edge")
        if other.outgoing:
            raise PipelineException("Can't connect to an outgoing "
                                    "edge")
        self.__other = other
        other.__other = self

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


class Node:

    """A real or virtual Node the Pipeline graph"""

    def __init__(self, stage, connections=None, label=None):
        """

        Parameters
        ----------
        stage: Stage
            The Stage to be run when this Node is executed
        connections: dict of {str : Connection} or None
            If provided, the Node will use the provided connections rather than
            making its own. This is useful for virtual nodes which manage
            Connections for a subgraph of the Pipeline rather than for an
            individual stage.
        label: str or None
            If provided, will be returned by this node's __str__ method. 
            Otherwise, will default to using __repr__

        """
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

    def __getitem__(self, key):
        """Gets the COnnections specified by key"""
        return self.__connections[key]

    def __repr__(self):
        return 'Node({})'.format(self.__stage)

    def __str__(self):
        if self.__label is None:
            return self.__repr__()
        return self.__label

    def get_stage(self):
        return self.__stage

    def get_inputs(self, live_only=True):
        """Returns a dictionary of {key : Connection} for all Connections
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
        """Returns a dictionary of {key : Connection} for all Connections
        that are outgoing.

        Parameters
        ----------
        live_only: bool
            if True, will only return Connections that are actually connected
                to another node.

        """
        return {key: self.__connections[key] for key in self.__connections
                if self.__connections[key].outgoing and
                (not live_only or self.__connections[key].other is not None)}


class Pipeline:

    """Internal representation of a UPSG pipeline.

    Our structure is merely a graph of pipeline elements. Execution will
    be relegated to either Drake or some simplified, internal replacement.

    """

    def __init__(self):
        self.__nodes = []

    def add(self, stage, label=None):
        """Add a stage to the pipeline

        Parameters
        ----------
        stage: an instance of Stage to add to the pipeline.
        label: str or None
            label to be returned by created Node's __str__ method. If not
            provieded, will use Node's __repr__ method

        Returns
        -------
        A Node encapsulating the given stage

        """
        # TODO this is here to avoid a circular import. Should refactor
        from stage import MetaStage, RunnableStage
        if isinstance(stage, RunnableStage):
            node = Node(stage, label=label)
            self.__nodes.append(node)
            return node
        if isinstance(stage, MetaStage):
            metanode = self.__integrate(*stage.pipeline)
            return metanode
        raise TypeError('Not a valid RunnableStage or MetaStage')

    def __integrate(self, other, in_node, out_node):
        """Integrates another pipeline into this one and creates a virtual
        uid to access the sub-pipeline.

        Parameters
        ----------
        other : Pipeline

        Returns
        -------
        Node which can be used to connect nodes to sub-pipeline as if
            the sub-pipeline were a single node.

        """
        self.__nodes += other.__nodes
        connections = {}
        connections.update(in_node.get_inputs(False))
        connections.update(out_node.get_outputs(False))
        return Node(None, connections=connections)

    def visualize(self, filename = None):
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

        returns
        -------
        potentially relative path of the rendered pdf

        """
        from graphviz import Digraph
        from os import system
        dot = Digraph()
        node_names = {}
        next_node_number = 0
        node_queue = [node for node in self.__nodes
                      if not node.get_outputs()]  # start with the root nodes
        for node in node_queue:
            name = 'node_{}'.format(next_node_number)
            node_names[node] = name
            dot.node(name, str(node), shape = 'box')
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
                    dot.node(other_name, str(other_node), shape = 'box')
                    node_names[other_node] = other_name
                dot.edge(other_name, name, label = '{}\n::\n{}'.format(
                    other_conn.key, conn.key))
            node_queue.extend(input_nodes)
            processed.add(node)
        out_file = dot.render(filename = filename)
        return out_file

    class Printer:
        def __init__(
                self, 
                fmt_doc_header='',
                fmt_doc_footer='',
                fmt_node_header='',
                fmt_node_footer='',
                fmt_arg_header='',
                fmt_arg_footer='',
                fmt_row_1='',
                fmt_row_2='',
                max_cols=2**32,
                str_cleanup=lambda s: s):
            if fmt_doc_header:
                self.__print_doc_header = lambda: print(fmt_doc_header)
            else:
                self.__print_doc_header = lambda: None
            if fmt_doc_footer:
                self.__print_doc_footer = lambda: print(fmt_doc_footer)
            else:
                self.__print_doc_footer = lambda: None
            if fmt_node_header:
                self.__print_node_header = lambda node: print(
                        fmt_node_header.format(node=str_cleanup(str(node))))
            else:
                self.__print_node_header = lambda node: None
            if fmt_node_footer:
                self.__print_node_footer = lambda node: print(
                        fmt_node_footer.format(node=str_cleanup(str(node))))
            else:
                self.__print_node_footer = lambda node: None
            if fmt_arg_header:
                self.__print_arg_header = lambda arrow, key, file_name: print(
                        fmt_arg_header.format(
                            arrow=str_cleanup(arrow), 
                            key=str_cleanup(key), 
                            file_name=str_cleanup(file_name)))
            else:
                self.__print_arg_header = lambda arrow, key, file_name: None
            if fmt_arg_footer:
                self.__print_arg_footer = lambda arrow, key, file_name: print(
                        fmt_arg_footer.format(
                            arrow=str_cleanup(arrow), 
                            key=str_cleanup(key), 
                            file_name=str_cleanup(file_name)))
            else:
                self.__print_arg_footer = lambda arrow, key, file_name: None
            if fmt_row_1 or fmt_row_2:
                self.__print_data = lambda a: print(self.__alternate_row_fmt(
                        a, 
                        fmt_row_1, 
                        fmt_row_2,
                        max_cols,
                        str_cleanup))
            else:
                self.__print_data = lambda a: None

        def __alternate_row_fmt(self, a, fmt_row_1, fmt_row_2, max_cols, str_cleanup):
            # http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
            # (2nd answer)
            header = fmt_row_1.format(
                    row=str_cleanup(','.join(a.dtype.names)[:max_cols]))
            if a.size <= 0:
                return header
            return '{}\n{}'.format(header, '\n'.join((fmt_row.format(
                row=str_cleanup(str(row)[:max_cols])) for 
                row, fmt_row in it.izip(
                    np.nditer(a), it.cycle((fmt_row_2, fmt_row_1))))))

        def header_print(self):
            self.__print_doc_header()

        def footer_print(self):
            self.__print_doc_footer()

        def stage_print(self, node, input_args, output_args):

            self.__print_node_header(node)

            for arg in input_args:
                self.__print_arg_header(
                        '<-', 
                        arg, 
                        input_args[arg].get_file_name())
                self.__print_data(input_args[arg].to_np())
                self.__print_arg_footer(
                        '<-',
                        arg,
                        input_args[arg].get_file_name())

            for arg in output_args:
                self.__print_arg_header(
                       '->',
                       arg,
                       output_args[arg].get_file_name())
                self.__print_data(output_args[arg].to_np())
                self.__print_arg_footer(
                       '->',
                       arg,
                       output_args[arg].get_file_name())

            self.__print_node_footer(node)

    def __get_ansi_stage_printer(self):
        # for colored debug printing
        # http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
        ANSI_HEADER_COLOR = '\x1b[30;42m'
        ANSI_FOOTER_COLOR = '\x1b[30;41m'
        ANSI_ROW_COLOR_1 = '\x1b[30;47m'
        ANSI_ROW_COLOR_2 = '\x1b[30;46m'
        ANSI_END = '\x1b[0m'
        fmt_node = '{}{{node}}{}'
        fmt_node_header = fmt_node.format(
            ANSI_HEADER_COLOR,
            ANSI_END)
        fmt_node_footer = fmt_node.format(
            ANSI_FOOTER_COLOR,
            ANSI_END)
        fmt_arg = '{}{{arrow}}{{key}}[{{file_name}}]:{}'
        fmt_arg_header = fmt_arg.format(
            ANSI_HEADER_COLOR,
            ANSI_END)
        fmt_arg_footer = fmt_arg.format(
            ANSI_FOOTER_COLOR,
            ANSI_END)
        fmt_row = '{}{{row}}{}'
        fmt_row_1 = fmt_row.format(ANSI_ROW_COLOR_1, ANSI_END)
        fmt_row_2 = fmt_row.format(ANSI_ROW_COLOR_2, ANSI_END)
        return self.Printer(
                '',
                '',
                fmt_node_header,
                fmt_node_footer,
                fmt_arg_header,
                fmt_arg_footer,
                fmt_row_1,
                fmt_row_2,
                80)

    def __get_bw_stage_printer(self):
        fmt_node = '{node}'
        fmt_arg = '{arrow}{key}[{file_name}]:'
        fmt_row = '{row}'
        return self.Printer( 
                '',
                '',
                fmt_node,
                '/' + fmt_node,
                fmt_arg,
                '/' + fmt_arg,
                fmt_row,
                fmt_row,
                80)

    def __get_html_stage_printer(self):
        # Note: Currently, this only works in Webkit
        doc_header = ('<!DOCTYPE html><html>'
                      '<head>'
                      '<style>'
                      'table td, th {'
                      '    border: 1px solid black;'
                      '}'
                      'table {'
                      '    border-collapse: collapse;'
                      '}'
                      'tr:nth-child(even) {'
                      '    background: cyan'
                      '}'
                      'tr:nth-child(odd) {'
                      '    background: white'
                      '}'
                      '</style>'
                      '</head>'
                      '<body>')
        doc_footer = '</body></html>'
        fmt_node_header = '<h1>{node}</h1>'
        fmt_node_footer = ''
        fmt_arg_header = ('<details>'
                          '<summary>{arrow}{key}[{file_name}]:</summary>'
                          '<table>')
        fmt_arg_footer = '</table></details>'
        fmt_row = '<tr><td>{row}</td></tr>'
        str_cleanup = lambda s: s.replace('<', '&lt').replace('>', '&gt')
        return self.Printer(
                doc_header,
                doc_footer,
                fmt_node_header,
                fmt_node_footer,
                fmt_arg_header,
                fmt_arg_footer,
                fmt_row,
                fmt_row,
                str_cleanup=str_cleanup)

    def run_debug(self, output='', single_step=False):
        """Run the pipeline in the current Python process.

        This method of running the job runs everything in serial on a single
        process. It is provided for debugging purposes for use with small jobs.
        For larger and more performant jobs, use the run method.
        """

        if output == 'color':
            stage_printer = self.__get_ansi_stage_printer()
        elif output == 'bw':
            stage_printer = self.__get_bw_stage_printer()
        elif output == 'html':
            stage_printer = self.__get_html_stage_printer()
        elif output == 'progress':
            stage_printer = self.Printer(fmt_node_header = 'completed: {node}',
                                         fmt_doc_footer='pipeline complete')
        else:
            stage_printer = self.Printer()

        stage_printer.header_print()
        # TODO what should the user call rather than run?
        node_queue = [node for node in self.__nodes
                      if not node.get_outputs()]  # start with the root nodes
        state = dict.fromkeys(self.__nodes, None)
        while node_queue:
            node = node_queue.pop()
            if state[node] is not None:  # already computed
                continue
            input_connections = node.get_inputs()
            input_nodes = frozenset([input_connections[input_key].other.node
                                     for input_key in input_connections])
            unfinished_dependencies = [dep_node for dep_node in input_nodes
                                       if state[dep_node] is None]
            if unfinished_dependencies:
                node_queue.append(node)
                node_queue += unfinished_dependencies
                continue
            input_args = {
                input_key: state[other][other_key] for input_key,
                other,
                other_key in map(
                    lambda k: (
                        k,
                        input_connections[k].other.node,
                        input_connections[k].other.key),
                    input_connections)}
            output_args = node.get_stage().run(node.get_outputs().keys(),
                                               **input_args)
            map(lambda k: output_args[k].write_to_read_phase(), output_args)
            stage_printer.stage_print(node, input_args, output_args)
            state[node] = output_args
            if single_step:
                pdb.set_trace()
        stage_printer.footer_print()

    def run(self, **kwargs):
        """Run the pipeline"""
        # TODO a better method of scheduling/running than this
        self.run_debug(**kwargs)
