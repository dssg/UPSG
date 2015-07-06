from __future__ import print_function

import abc
import os
from HTMLParser import HTMLParser
import itertools as it
from StringIO import StringIO

import numpy as np

from .utils import html_escape
from .utils import np_to_html_table
from .utils import html_format
from .uobject import UObjectException

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
            self.__print_arg_header = lambda arrow, key: print(
                    fmt_arg_header.format(
                        arrow=str_cleanup(arrow), 
                        key=str_cleanup(key)))
        else:
            self.__print_arg_header = lambda arrow, key: None
        if fmt_arg_footer:
            self.__print_arg_footer = lambda arrow, key: print(
                    fmt_arg_footer.format(
                        arrow=str_cleanup(arrow), 
                        key=str_cleanup(key)))
        else:
            self.__print_arg_footer = lambda arrow, key: None
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
            a = uo.to_np()[:10]
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
                    arg)
            self.__print_data(input_args[arg])
            self.__print_arg_footer(
                    '<-',
                    arg)

        for arg in output_args:
            self.__print_arg_header(
                   '->',
                   arg)
            self.__print_data(output_args[arg])
            self.__print_arg_footer(
                   '->',
                   arg)

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
        return html_format(fmt, *args, **kwargs)

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
        np_to_html_table(a, self.__fout)

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
                    '</b></summary>\n'),
                   node[arg].edge.uid,
                   arg,
                   node[arg].other.key,
                   node[arg].other.node.uid,
                   node[arg].other.node))
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

def get_ansi_stage_printer():
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
    fmt_arg = '{}{{arrow}}{{key}}:{}'
    fmt_arg_header = fmt_arg.format(
        ANSI_HEADER_COLOR,
        ANSI_END)
    fmt_arg_footer = fmt_arg.format(
        ANSI_FOOTER_COLOR,
        ANSI_END)
    fmt_row = '{}{{row}}{}'
    fmt_row_1 = fmt_row.format(ANSI_ROW_COLOR_1, ANSI_END)
    fmt_row_2 = fmt_row.format(ANSI_ROW_COLOR_2, ANSI_END)
    return Printer(
            '',
            '',
            fmt_node_header,
            fmt_node_footer,
            fmt_arg_header,
            fmt_arg_footer,
            fmt_row_1,
            fmt_row_2,
            80)

def get_bw_stage_printer():
    fmt_node = '{node}'
    fmt_arg = '{arrow}{key}:'
    fmt_row = '{row}'
    return Printer( 
            '',
            '',
            fmt_node,
            '/' + fmt_node,
            fmt_arg,
            '/' + fmt_arg,
            fmt_row,
            fmt_row,
            80)

def get_progress_stage_printer():    
    return Printer(fmt_node_header = 'completed: {node}',
                        fmt_doc_footer='pipeline complete')

DEBUG_OUTPUT_ENV_VAR = 'UPSG_DEBUG_OUTPUT_MODE'

def run(pipeline, nodes, output='', report_path='', single_step=False):
    """Run the pipeline in the current Python process.

    This method of running the job runs everything in serial on a single
    process. It is provided for debugging purposes for use with small jobs.
    For larger and more performant jobs, use the run method.

    Parameters
    ----------
    pipeline : upsg.pipeline.Pipeline
        The Pipeline to run
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

    if output == '':
        try:
            output = os.environ[DEBUG_OUTPUT_ENV_VAR]
        except KeyError:
            output = 'silent'
    if output == 'color':
        stage_printer = get_ansi_stage_printer()
    elif output == 'bw':
        stage_printer = get_bw_stage_printer()
    elif output == 'progress':
        stage_printer = get_progress_stage_printer()
    elif output == 'html':
        if not report_path:
            report_path = 'graph_out.html'
        print('Generating graph visualization')
        png_file, map_file, scale = pipeline.visualize(html_map=True)
        stage_printer = GraphPrinter(
                png_file, 
                map_file, 
                scale,
                report_path,
                get_progress_stage_printer())
        print('Visualization complete')
    else:
        stage_printer = Printer()

    stage_printer.header_print()
    # TODO what should the user call rather than run?
    node_queue = [node for node in nodes
                  if not node.get_outputs()]  # start with the root nodes
    state = dict.fromkeys(nodes, None)
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
            import pdb
            pdb.set_trace()
    stage_printer.footer_print()
