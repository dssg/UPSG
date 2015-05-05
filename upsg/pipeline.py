from __future__ import print_function
from collections import namedtuple
from StringIO import StringIO
from HTMLParser import HTMLParser
import os
import weakref
import uuid
import abc
import subprocess
import re
import cgi
import itertools as it
import numpy as np

from .uobject import UObjectException
from .utils import html_escape


class PipelineException(Exception):
    pass


class Edge(object):
    """A directed graph edge"""

    def __init__(self, conn_from, conn_to, uid=None):
        """

        Parameters
        ----------
        conn_from: Connection
            The Connection which the edge is directed from

        conn_to: Connection
            The Connection which the edge is directed to

        uid: str or None
            The unique uid of edge. If not provided, a uid will be generated

        """
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

    """A real or virtual Node the Pipeline graph"""

    def __init__(self, stage, connections=None, label=None, uid=None):
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
        uid: str or None
            The unique uid of edge. If not provided, a uid will be generated

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
        if uid == None:
            uid = 'node_{}'.format(uuid.uuid4())
        self.__uid = uid    

    def __getitem__(self, key):
        """Gets the COnnections specified by key"""
        return self.__connections[key]

    def __call__(self, *args, **kwargs):
        """Alternative syntax for connecting Stages together

        Parameters
        ----------
        args : list of (Node or Connection)
            The nth output_key of self.output_keys will be connected to the
            nth node or connection. For example, if self.output_keys == 
            ['out', 'complement', 'status'], and we invoke 
            self(clf_node['X_train'], clf_node['X_test'], status_node),
            it is equivalent to doing:

            self['out'] > clf_node['X_train'] 
            self['complement'] > clf_node['X_test']
            self['status'] > status_node

        kwargs : dict of str: (Node or Connection)
            The output key corresponding to the given keyword will be 
            connected to the argument assigned to that output key. For
            example, if we invoke self(complement=clf_node['X_test'], 
            status=status_node, out=clf_node['X_train']) it is equivalent to
            doing:

            self['complement'] > clf_node['X_test']
            self['status'] > status_node
            self['out'] > clf_node['X_train'] 

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
    be relegated to either Drake or some simplified, internal replacement.

    """

    def __init__(self):
        self.__nodes = []

    def __struct_str_rep(self, pipeline):
        return {str(node) : {key: (str(conn.other.node), conn.other.key) for
                             key, conn in node.get_outputs().items()}
                for node in pipeline.__nodes}

    def is_equal_by_str(self, other):
        """Returns whether or not self has the same nodes and edges as 
        Pipeline other. Node equality is determined by whether nodes have the
        same str representation, so it's really only useful in contrived 
        circumstances like our unit tests"""
        return self.__struct_str_rep(self) == self.__struct_str_rep(other)
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
        Node which can be used to connect nodes to sub-pipeline as if
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

        returns
        -------
        
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

    class BasePrinter(object):
        __metaclass__ = abc.ABCMeta
        @abc.abstractmethod
        def header_print(self):
            pass

        @abc.abstractmethod
        def footer_print(self):
            pass

        @abc.abstractmethod
        def stage_print(self, node, input_args, output_args):
            pass

    class Printer(BasePrinter):
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
                self.__print_data = lambda uo: print(self.__alternate_row_fmt(
                        uo, 
                        fmt_row_1, 
                        fmt_row_2,
                        max_cols,
                        str_cleanup))
            else:
                self.__print_data = lambda a: None

        def __alternate_row_fmt(self, uo, fmt_row_1, fmt_row_2, max_cols, str_cleanup):
            # http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
            # (2nd answer)
            try:
                a = uo.to_np()
            except UObjectException: # unsupported conversion
                return ''
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
                self.__print_data(input_args[arg])
                self.__print_arg_footer(
                        '<-',
                        arg,
                        input_args[arg].get_file_name())

            for arg in output_args:
                self.__print_arg_header(
                       '->',
                       arg,
                       output_args[arg].get_file_name())
                self.__print_data(output_args[arg])
                self.__print_arg_footer(
                       '->',
                       arg,
                       output_args[arg].get_file_name())

            self.__print_node_footer(node)

    class GraphPrinter(BasePrinter):
        def __init__(self, png_file, map_file, scale, out_file, term_printer):
            self.__out_file = out_file
            self.__fout = open(out_file, 'w')
            with open(map_file) as f_map_file:
                html_map = f_map_file.read()
            map_parser = self.MapParser(scale)
            map_parser.feed(html_map)
            self.__map_id = map_parser.map_id
            self.__map = map_parser.get_map()
            self.__png_file = png_file
            self.__term_printer = term_printer

        def __del__(self):
            self.__fout.close()

        class MapParser(HTMLParser):
            def __init__(self, scale):
                HTMLParser.__init__(self)
                self.__scale = scale
                self.__sio = StringIO()
                self.map_id = ''
            def handle_starttag(self, tag, attrs):
                sio = self.__sio
                scale = self.__scale
                sio.write('<{} '.format(tag))
                for name, value in attrs:
                    if name=='coords':
                        value = ','.join(
                                [str(int(float(coord) * scale)) for
                                 coord in value.split(',')])
                    elif name=='id' and tag=='map':
                        self.map_id = value
                    value = html_escape(value)
                    sio.write('{}="{}" '.format(name, value))
                sio.write('>')
            def handle_endtag(self, tag):
                self.__sio.write('</{}>'.format(tag))
            def handle_data(self, data):
                self.__sio.write(data)
            def handle_entityref(self, name):
                self.__sio.write('&{};'.format(name))
            def handle_charref(self, name):
                self.__sio.write('&#{};'.format(name))
            def get_map(self):
                return self.__sio.getvalue()

        def __html_format(self, fmt, *args, **kwargs):
            clean_args = [html_escape(str(arg)) for arg in args]
            clean_kwargs = {key: html_escape(str(kwargs[key])) for 
                            key in kwargs}
            return fmt.format(*clean_args, **clean_kwargs)

        def header_print(self):
            doc_header = ('<!DOCTYPE html>\n'
                          '<html>\n'
                          '<head>\n'
                          '<style>\n'
                          ':target {\n'
                          '    background-color: yellow;\n'
                          '}\n'
                          'table td, th {\n'
                          '    border: 1px solid black;\n'
                          '}\n'
                          'table {\n'
                          '    border-collapse: collapse;\n'
                          '}\n'
                          'tr:nth-child(even) {\n'
                          '    background: cyan\n'
                          '}\n'
                          'tr:nth-child(odd) {\n'
                          '    background: white\n'
                          '}\n'
                          '</style>\n'
                          '</head>\n'
                          '<body>\n')
            self.__fout.write(doc_header)
            self.__fout.write(self.__map)
            self.__fout.write(self.__html_format(
                '<div id=top><img src="{}" usemap="#{}"/></div>\n',
                self.__png_file, 
                self.__map_id))
            self.__term_printer.header_print()

        def footer_print(self):
            self.__fout.write('</body>\n</html>')
            self.__term_printer.footer_print()
            print('Report printed to: {}'.format(
                os.path.abspath(self.__out_file)))

        def __data_print(self, a):
            self.__fout.write('<p>table of shape: ({},{})</p>'.format(
                len(a),
                len(a.dtype)))
            self.__fout.write('<p><table>\n')
            header = '<tr>{}</tr>\n'.format(
                ''.join(
                        [self.__html_format(
                            '<th>{}</th>',
                            name) for 
                         name in a.dtype.names]))
            self.__fout.write(header)
            rows = a[:100]
            data = '\n'.join(
                ['<tr>{}</tr>'.format(
                    ''.join(
                        [self.__html_format(
                            '<td>{}</td>',
                            cell) for
                         cell in row])) for
                 row in rows])
            self.__fout.write(data)
            self.__fout.write('\n')
            self.__fout.write('</table></p>')

        def stage_print(self, node, input_args, output_args):

            self.__fout.write(self.__html_format(
                '<p><div id={}><h3>{}</h3>\n',
                node.uid,
                node))

            for arg in output_args:
                if not node[arg].other: 
                    # This output isn't connected to anything. Ignore it.
                    continue
                self.__fout.write(self.__html_format(
                       ('<p><div id={}><details><summary>'
                        '<b>&rarr;{}::{}&rarr;<a href=#{}>{}</a> '
                        '<a href={}>[{}]</a><b></summary>\n'),
                       node[arg].edge.uid,
                       arg,
                       node[arg].other.key,
                       node[arg].other.node.uid,
                       node[arg].other.node,
                       output_args[arg].get_file_name(),
                       output_args[arg].get_file_name()))
                uo = output_args[arg]
                try:
                    a = uo.to_np()
                    self.__data_print(a)
                except UObjectException:
                    try:
                        a = uo.to_external_file()
                        self.__fout.write(self.__html_format(
                            '<p>external file: <a href={}>{}</a></p>', 
                            a,
                            a))
                    except UObjectException:
                        pass
                self.__fout.write('</details></div></p>\n')
            self.__fout.write('<p><a href=#top>[TOP]</a></p></div></p>\n')
            self.__term_printer.stage_print(node, input_args, output_args)

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

    def __get_progress_stage_printer(self):    
        return self.Printer(fmt_node_header = 'completed: {node}',
                            fmt_doc_footer='pipeline complete')

    def run_debug(self, output='', report_path='', single_step=False):
        """Run the pipeline in the current Python process.

        This method of running the job runs everything in serial on a single
        process. It is provided for debugging purposes for use with small jobs.
        For larger and more performant jobs, use the run method.

        Parameters
        ----------
        output: str
            Method of displaying output. One of:
                'bw': prints progress and truncated stage output to terminal
                'color': prints progress and truncated stage output 
                         to terminal using ANSI colors
                'progress': only prints progress
                'html': prints pipeline visualization and truncated output
                        in an html report. Also prints progress to terminal
                'silent' or unspecified: prints no output.
        
        report_path: str
            If output='html', the path of the html file to be generated.
            If unspecified, will use graph_out.html in the current working
            directory

        single_step: bool
            If True, will invoke pdb after every stage is run

        """

        if output == 'color':
            stage_printer = self.__get_ansi_stage_printer()
        elif output == 'bw':
            stage_printer = self.__get_bw_stage_printer()
        elif output == 'progress':
            stage_printer = self.__get_progress_stage_printer()
        elif output == 'html':
            if not report_path:
                report_path = 'graph_out.html'
            print('Generating graph visualization')
            png_file, map_file, scale = self.visualize(html_map=True)
            stage_printer = self.GraphPrinter(
                    png_file, 
                    map_file, 
                    scale,
                    report_path,
                    self.__get_progress_stage_printer())
            print('Visualization complete')
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
                    input_connections) if other_key in state[other]}
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
