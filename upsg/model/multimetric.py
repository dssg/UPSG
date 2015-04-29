from collections import namedtuple
import uuid

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase
from ..pipeline import Pipeline
from ..utils import dict_to_np_sa
from ..wrap.wrap_sklearn import wrap, wrap_and_make_instance
from ..export.plot import Plot
from ..transform.identity import Identity

VisualMetricSpec = namedtuple('VisualMetricSpec', ['metric', 
                                                   'output_key_x',
                                                   'output_key_y',
                                                   'graph_title',
                                                   'graph_x_label',
                                                   'graph_y_label'])


NumericMetricSpec = namedtuple('NumericMetricSpec', ['metric',
                                                     'output_key',
                                                     'title'])

class Multimetric(MetaStage):

    class __ReduceStage(RunnableStage):

        def __init__(self, metrics, title, file_name):
            self.__file_name = file_name
            self.__title = title
            self.__n_metrics = len(metrics)
            self.__metrics = metrics
            self.__input_keys = (['params'] + 
                                 ['metric{}_in'.format(i) for 
                                  i in xrange(self.__n_metrics)])   

            self.__output_keys = ['report_file']

        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return self.__output_keys


        def run(self, outputs_requested, **kwargs):
            # TODO sanitize html
            # TODO use dbg printer's table printing
            with open(self.__file_name, 'w') as fout:
                fout.write(
                        '<h3>{}</h3><h4>Best params</h4>\n<p>{}</p>\n'.format(
                            self.__title, 
                            kwargs['params'].to_dict()))
                for i, metric in enumerate(self.__metrics):
                    uo = kwargs['metric{}_in'.format(i)]
                    if isinstance(metric, VisualMetricSpec):
                        fout.write(
                            '<h4>{}</h4><p><img src="{}"/></p>\n'.format(
                                    metric.graph_title,
                                    uo.to_external_file()))
                    else:
                        fout.write(
                            '<h4>{}</h4>\n<p>{}</p>\n'.format(
                                    metric.title,
                                    uo.to_np()))

            uo_report_file = UObject(UObjectPhase.Write)
            uo_report_file.from_external_file(self.__file_name)
            return {'report_file': uo_report_file}

    def __init__(self, metrics, title, file_name=None):

        p = Pipeline()
        self.__pipeline = p
        if file_name is None:
            file_name = str(uuid.uuid4()) + '.html'

        self.__file_name = file_name

        node_map = p.add(Identity(('params', 'pred_proba', 'y_true')))
        node_reduce = p.add(self.__ReduceStage(metrics, title, file_name))

        node_map['params_out'] > node_reduce['params']

        for i, metric in enumerate(metrics):
            stage_metric = wrap_and_make_instance(metric.metric)
            in_keys = stage_metric.input_keys
            # TODO develop this to support more metrics. Maybe share code w/
            # test_wrap
            node_metric = p.add(stage_metric)
            if 'y_true' in in_keys:
                node_map['y_true_out'] > node_metric['y_true']
            if 'y_score' in in_keys:
                node_map['pred_proba_out'] > node_metric['y_score']
            if 'probas_pred' in in_keys:
                node_map['pred_proba_out'] > node_metric['probas_pred']

            metric_in_key = 'metric{}_in'.format(i)
            if isinstance(metric, VisualMetricSpec):
                out_file = '{}.png'.format(uuid.uuid4())
                node_plot = p.add(Plot(
                    out_file,
                    xlabel = metric.graph_x_label,
                    ylabel = metric.graph_y_label))
                node_metric[metric.output_key_x] > node_plot['x']
                node_metric[metric.output_key_y] > node_plot['y']
                node_plot['plot_file'] > node_reduce[metric_in_key]
            else:
                node_metric[metric.output_key] > node_reduce[metric_in_key]

        self.__in_node = node_map
        self.__out_node = node_reduce

    @property
    def input_keys(self):
        return self.__in_node.input_keys

    @property
    def output_keys(self):
        return self.__out_node.output_keys

    @property
    def pipeline(self):
        return (self.__pipeline, self.__in_node, self.__out_node)
