import tokenize
from StringIO import StringIO
from token import *
import itertools as it
import numpy as np
import ast

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold as SKKFold

from ..stage import RunnableStage
from ..uobject import UObject, UObjectPhase

class SplitColumns(RunnableStage):
    """Splits a table 'in' into two tables 'selected' and 'rest' where 
    'selected' consists of the given columns and 'rest' consists of the rest
    of the columns"""

    def __init__(self, columns):
        """

        parameters
        ----------
        columns: list of str
            Colums that will appear in the 'selected' table but not the 'rest'
            table
        """
        self.__columns = columns

    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return ['selected', 'rest']

    def run(self, outputs_requested, **kwargs):
        # TODO different implementation if internally sql?
        columns = list(self.__columns)

        to_return = {}
        in_array = kwargs['in'].to_np()

        if 'selected' in outputs_requested:
            uo_selected = UObject(UObjectPhase.Write)
            uo_selected.from_np(in_array[columns])
            to_return['selected'] = uo_selected

        if 'rest' in outputs_requested:
            uo_rest = UObject(UObjectPhase.Write)
            # http://stackoverflow.com/questions/3462143/get-difference-between-two-lists
            remaining_columns = list(set(in_array.dtype.names) - set(columns))
            uo_rest.from_np(in_array[remaining_columns])
            to_return['rest'] = uo_rest

        return to_return


class SplitColumn(RunnableStage):

    """Splits a table 'in' into two tables 'X' and 'y' where y is one column of
    A and X is everything else. """

    def __init__(self, column):
        """

        parameters
        ----------
        column: int or str
            index or name of the column from which 'y' will be extracted

        """
        self.__column = column

    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return ['X', 'y']

    def run(self, outputs_requested, **kwargs):
        uo_X = UObject(UObjectPhase.Write)
        uo_y = UObject(UObjectPhase.Write)
        in_array = kwargs['in'].to_np()
        names = list(in_array.dtype.names)
        if isinstance(self.__column, int):
            col_name = names[self.__column]
        else:
            col_name = self.__column
        uo_y.from_np(in_array[[col_name]])
        names.remove(col_name)
        uo_X.from_np(in_array[names])
        return {'X': uo_X, 'y': uo_y}


class SplitTrainTest(RunnableStage):

    """
    
    Splits tables 'in0', 'in1', 'in2', ... into training and testing
    data 'train0', 'test0', 'train1', 'test1', 'train2', 'test2', ...

    All input tables should have the same number of rows
    
    """
    # TODO wrap.wrap_sklearn in a more general way, like in wrap.wrap_sklearn
    # TODO split more than one array at a time

    def __init__(self, n_arrays=1, **kwargs):
        """

        parameters
        ----------
        n_arrays: int (default 1)
            the number of arrays that will be split
        kwargs:
            arguments corresponding to the keyword arguments of
            sklearn.cross_validation.train_test_split

        """
        self.__kwargs = kwargs
        self.__n_arrays = n_arrays

        self.__input_keys = map('in{}'.format, xrange(n_arrays))
        self.__output_keys = (map('train{}'.format, xrange(n_arrays)) +
                              map('test{}'.format, xrange(n_arrays)))

    @property
    def input_keys(self):
        return self.__input_keys

    @property
    def output_keys(self):
        return self.__output_keys

    def run(self, outputs_requested, **kwargs):
        in_arrays = [kwargs[key].to_np() for key in self.__input_keys]
        splits = train_test_split(*in_arrays, **self.__kwargs)
        results = {key: UObject(UObjectPhase.Write) for key
                   in self.__output_keys}
        for index, in_key in enumerate(self.__input_keys):
            key_number = int(in_key.replace('in', ''))
            results['train{}'.format(key_number)].from_np(splits[2 * index])
            results['test{}'.format(key_number)].from_np(splits[2 * index + 1])
        return results


class KFold(RunnableStage):
    """
    
    Splits tables 'in0', 'in1', 'in2', ... into n_folds train and test sets 
    called:
        'train0_0', 'test0_0', 'train0_1', 'test0_1',... (corresponding to
        different folds of 'in0')
        'train1_0', 'test1_0', 'train1_1', 'test1_1',... (corresponding to
        different folds of 'in1')
        'train2_0', 'test2_0', 'train2_1', 'test2_1',... (corresponding to
        different folds of 'in2')
        ...

    All input tables should have the same number of rows

    """

    def __init__(self, n_arrays=1, n_folds=2, **kwargs):
        """

        parameters
        ----------
        n_arrays: int (default 1)
            The number of arrays that will be split
        n_folds: int (default 2)
            The number of folds. Must be at least 2.
        kwargs:
            Arguments corresponding to the keyword arguments of
            sklearn.cross_validation.KFold other than n and
            n_folds

        """
        self.__kwargs = kwargs
        self.__n_arrays = n_arrays
        self.__n_folds = n_folds

        self.__input_keys = ['in{}'.format(array) for array in 
                             xrange(n_arrays)]
        self.__output_keys = list(it.chain.from_iterable(
                (('train{}_{}'.format(array, fold), 
                  'test{}_{}'.format(array, fold))
                 for array, fold in it.product(
                     xrange(n_arrays), xrange(n_folds)))))

    @property
    def input_keys(self):
        return self.__input_keys

    @property
    def output_keys(self):
        return self.__output_keys

    def run(self, outputs_requested, **kwargs):
        in_arrays = [kwargs[key].to_np() for key in self.__input_keys]
        if len(in_arrays) < 1:
            return {}
        kf = SKKFold(in_arrays[0].shape[0], self.__n_folds, **self.__kwargs)
        results = {key: UObject(UObjectPhase.Write) for key
                   in self.__output_keys}
        for fold_index, (train_inds, test_inds) in enumerate(kf):
            for array_index, in_key in enumerate(self.__input_keys):
                key_number = int(in_key.replace('in', ''))
                results['train{}_{}'.format(key_number, fold_index)].from_np(
                    in_arrays[array_index][train_inds])
                results['test{}_{}'.format(key_number, fold_index)].from_np(
                    in_arrays[array_index][test_inds])
        return results

class QueryError(Exception):
    pass

class Query(RunnableStage):
    """Selects rows to put in a table based on a given query

    Input Keys
    ----------
    in

    Ouptu Keys
    ----------
    out: table containing only rows that match the query
    out_inds: one-column table containing the indices of in that matched the query
    complement: table containing only rows that do not match the query
    complement_inds: one-column table containing the indices of in that did not match the query
    
    """
    __IN_TABLE_NAME = 'in_table'

    class __QueryParser(ast.NodeTransformer):
        # TODO enforce the constraint that some object somewhere has to be a
        # column name

        BINARY_OPS = {
            ast.Or: 'np.logical_or',
            ast.And: 'np.logical_and'
        }

        UNARY_OPS = {
            ast.Not: 'np.logical_not'
        }

        SUPPORTED_CMP = [
            ast.Eq, 
            ast.NotEq, 
            ast.Lt, 
            ast.LtE,
            ast.Gt,
            ast.GtE]

        def __init__(self, col_names, array_name):
            self.__col_names = col_names
            self.__array_name = ast.Name(id=array_name, ctx=ast.Load())

        def __visit_op(self, np_op, *args):
            module, attr = np_op.split('.')
            func = ast.Attribute(
                    value=ast.Name(
                        id=module,
                        ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load())
            call_args = [self.visit(arg) for arg in args]
            return ast.Call(
                    func=func, 
                    args=call_args,
                    keywords=[],
                    starargs=None,
                    kwargs=None)

        def visit_Expression(self, node):
            return ast.Expression(self.visit(node.body))

        def visit_Expr(self, node):
            return ast.Expr(value=self.visit(node.value))
            
        def visit_BoolOp(self, node):
            try:
                np_op = self.BINARY_OPS[type(node.op)]
            except KeyError:
                raise QueryError('BoolOp {} not supported'.format(node.op))
            return self.__visit_op(np_op, *node.values)

        def visit_UnaryOp(self, node):
            try:
                np_op = self.UNARY_OPS[type(node.op)]
            except KeyError:
                raise QueryError('UnOp {} not supported'.format(node.op))
            return self.__visit_op(np_op, node.operand)

        def visit_Compare(self, node):
            if len(node.ops) > 1:
                raise QueryError('cascading comparisons (e.g. 1 < 2 < 3) not'
                                 ' supported')
            op = node.ops[0]
            if type(op) not in self.SUPPORTED_CMP:
                raise QueryError('Compare op {} not supported'.format(op))
            return ast.Compare(
                    left=self.visit(node.left),
                    ops=node.ops,
                    comparators=[self.visit(comp) for comp in 
                                 node.comparators])

        def visit_Name(self, node):
            col_name = node.id
            if col_name not in self.__col_names:
                raise QueryError('\'{}\' is not a valid column name'.format(
                    col_name))
            sub_slice = ast.Index(value=ast.Str(s=col_name)) 
            return ast.Subscript(
                    value=self.__array_name, 
                    slice=sub_slice,
                    ctx=ast.Load())

        def visit_Str(self, node):
            return node

        def visit_Num(self, node):
            return node

        def visit_Call(self, node):
            func = node.func
            if not isinstance(func, ast.Name) or func.id != 'DT':
                raise QueryError('The only supported function call is DT()')
            return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(
                            id='np',
                            ctx=ast.Load()),
                        attr='datetime64',
                        ctx=ast.Load()),
                    args=node.args,
                    keywords=[],
                    starargs=None,
                    kwargs=None)

        def generic_visit(self, node):
            raise QueryError('node {} not supported'.format(node))

    def __init__(self, query):
        """

        Parameters
        ----------
        query : str
            A query used to select rows conforming to a small, Python-like
            langauge defined as follows:

            primary_expr: 
                '(' expr ')' | 
                expr
            expr: 
                mask | 
                unop primary_expr | 
                binop_expr
            binop_expr:
                '(' expr ')' binop '(' expr ')' |
                '(' expr ')' binop variable |
                variable binop '(' expr ')' 
            variable: 
                literal | 
                col_name | 
                '(' literal ')' | 
                '(' col_name ')'
            mask: 
                col_name binop col_name | 
                col_name binop literal |
                literal binop col_name |
                col_name -- for columns of boolean type
            unop : 'not'
            binop: '>' | '>=' | '<' | '<=' | '==' | '!=' | 'and' | 'or'
            literal: 
                NUMBER | 
                STRING |
                datetime
            datetime:
                "DT('" ISO_8601_DATE_OR_DATETIME_STRING "')" 

            col_names need to be the name of a column in the table. col_names
            SHOULD NOT be quoted. Literal string SHOULD be quoted

        examples
        --------
        >>> q1 = Query("id > 50")
        >>> q2 = Query("(name == 'Sarah') and (salary > 50000)")
        >>> q3 = Query("(start_dt != end_dt) or (not category == 2")
        >>> q4 = Query("start_dt < DT('2014-06-01')")

        """
        self.__query = query


    @property
    def input_keys(self):
        return ['in']

    @property
    def output_keys(self):
        return ['out', 'complement', 'out_inds', 'complement_inds']

    def __get_ast(self, col_names):
        parser = self.__QueryParser(col_names, self.__IN_TABLE_NAME)
        query = ast.fix_missing_locations(
                parser.visit(ast.parse(self.__query, mode='eval')))
        return query

    def dump_ast(self, col_names):
        """Dumps the AST of the query transformed into Python. Provided for debugging purposes."""
        query = self.__get_ast(col_names)
        return ast.dump(query)

    def run(self, outputs_requested, **kwargs):
        # TODO find some interface that doesn't involve string parsing
        # modeled after pandas.Dataframe.query:
        #     http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.query.html
        # which implements its own computation engine:
        #     http://pandas.pydata.org/pandas-docs/dev/generated/pandas.eval.html
        # supports numpy arithmetic comparison operators:
        #     http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#arithmetic-and-comparison-operations
        in_table = kwargs['in'].to_np()
        col_names = in_table.dtype.names
        query = self.__get_ast(col_names)
        mask = eval(compile(query, '<string>', 'eval'))
        ret = {}
        if 'out' in outputs_requested:
            uo_out = UObject(UObjectPhase.Write)
            uo_out.from_np(in_table[mask])
            ret['out'] = uo_out
        if 'complement' in outputs_requested:
            uo_comp = UObject(UObjectPhase.Write)
            uo_comp.from_np(in_table[np.logical_not(mask)])
            ret['complement'] = uo_comp
        if 'out_inds' in outputs_requested:
            uo_out_inds = UObject(UObjectPhase.Write)
            uo_out_inds.from_np(np.where(mask)[0])
            ret['out_inds'] = uo_out_inds
        if 'complement_inds' in outputs_requested:
            uo_comp_inds = UObject(UObjectPhase.Write)
            uo_comp_inds.from_np(np.where(np.logical_not(mask))[0])
            ret['complement_inds'] = uo_comp_inds
        return ret

