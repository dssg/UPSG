import numpy as np

from numpy.lib.recfunctions import merge_arrays

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase
from ..utils import np_sa_to_nd
from ..pipeline import Pipeline
from .identity import Identity
from .split import SplitColumns
from .lambda_stage import LambdaStage
from .rename_cols import RenameCols


class GenerateFeature(MetaStage):

    """
    Applies a given function only to the selected columns and outputs a 
    table consisting only of the columns with the function applied.

    **Input Keys**

    input
        table from which to generate features
    
    **Output Keys**

    output
        table with generated colume

    Parameters
    ----------
    gen_func : numpy structured array -> numpy structured array
        Function to generate features. The input array will consist of the
        columns selected. The output array should have the generated feature
    in_col_names : list of str
        Names of the columns to which the transform Stage will be supplied
    out_col_names : list of str or None
        Names of columns for generated features. If None, names will be 
        automatically generated

    """

    def __init__(self, gen_func, in_col_names, out_col_names=None):
        p = Pipeline()
        self.__pipeline = p

        in_node = p.add(Identity(input_keys=['input']))

        split_node = p.add(SplitColumns(in_col_names))
        split_node(in_node)

        lambda_node = p.add(LambdaStage(gen_func, n_outputs=1))
        lambda_node(split_node)

        out_node = p.add(Identity(output_keys=['output']))

        if out_col_names is not None:
            rename = p.add(RenameCols(out_col_names))
            rename(lambda_node)
            out_node(rename)
        else:
            out_node(lambda_node)

        self.__pipeline = p
        self.__in_node = in_node
        self.__out_node = out_node

    @property
    def input_keys(self):
        return self.__in_node.input_keys

    @property
    def output_keys(self):
        return self.__out_node.output_keys

    @property
    def pipeline(self):
        return (self.__pipeline, self.__in_node, self.__out_node)
