from collections import namedtuple
import logging

import numpy as np

import luigi
import luigi.mock

from .uobject import UObject, UObjectPhase
from .utils import get_resource_path

def node_to_task(node, context):
    logger = logging.getLogger('luigi-interface')

    # we need to keep track of which node gives which output
    node_inputs = node.get_inputs()
    node_outputs = node.get_outputs()
    req_tasks = {key: context[conn.other.node] for
                 key, conn in node_inputs.iteritems()}
    def requires(self):
        return req_tasks

    # TODO nonlocal targets
    out_files = {key: luigi.file.LocalTarget('{}.upsg'.format(
                    conn.edge.uid)) for key, conn  in node_outputs.iteritems()}
    def output(self):
        return out_files

    others_output_keys = {in_key: node_inputs[in_key].other.key 
                          for in_key in node_inputs}
    def run(self):
        logger.debug('running UPSG node: {} (uid {}): '
                     '#in_keys# {} #out_keys# {}'.format(
                        node, 
                        node.uid, 
                        node_inputs.keys(),
                        node_outputs.keys()))
        input_files = {in_key:
                       self.input()[in_key][others_output_keys[in_key]].open('r') 
                       for in_key in others_output_keys}
        input_args = {in_key:
                      UObject(
                          UObjectPhase.Read, 
                          input_files[in_key].read()) for in_key in 
                          others_output_keys}
        [input_files[in_key].close() for in_key in input_files]
        expected_out_keys = node_outputs.keys()
        output_args = node.get_stage().run(
                expected_out_keys, 
                **input_args)

        actual_out_keys = output_args.keys()  
        for key in expected_out_keys:
            if key not in actual_out_keys:
                logger.warning(
                        ('Expected key {} not returned by node {}. '
                         'Providing empty table'.format(
                            key,
                            node)))
                uo = UObject(UObjectPhase.Write)
                uo.from_np(np.array([]))
                output_args[key] = uo

        output_files = {out_key: self.output()[out_key].open('w') for 
                        out_key in node_outputs}
        [output_files[out_key].write(output_args[out_key].get_image()) for 
         out_key in output_files]
        [output_files[out_key].close() for out_key in output_files]
        self.__complete = True
        [input_args[in_key].cleanup() for in_key in input_args]
        [output_args[out_key].cleanup() for out_key in output_args]
        

    def __init__(self):
        luigi.Task.__init__(self)
        self.__complete = False

    def complete(self):
        return self.__complete

    task = type(
        'Task_{}'.format(node.uid), 
        (luigi.Task,), 
        {'__init__': __init__,
         'requires': requires, 
         'output': output, 
         'run': run, 
         'complete': complete})()


    context[node] = task
    return task


def run(nodes, logging_conf_file=None, worker_processes=None):
    node_queue = [node for node in nodes
                  if not node.get_outputs()]  # start with the leaves
    context = dict.fromkeys(nodes, None)
    luigi.interface.setup_interface_logging(logging_conf_file)
    sch = luigi.scheduler.CentralPlannerScheduler()
    if worker_processes is None:
        import multiprocessing
        worker_processes = multiprocessing.cpu_count()
    #w = luigi.worker.Worker(scheduler=sch, worker_processes=worker_processes)
    w = luigi.worker.Worker(scheduler=sch)
    while node_queue:
        node = node_queue.pop()
        if context[node] is not None:  # already computed
            continue
        input_connections = node.get_inputs()
        input_nodes = frozenset([input_connections[input_key].other.node
                                 for input_key in input_connections])
        unfinished_dependencies = [dep_node for dep_node in input_nodes
                                   if context[dep_node] is None]
        if unfinished_dependencies:
            node_queue.append(node)
            node_queue += unfinished_dependencies
            continue
        w.add(node_to_task(node, context))
    w.run()        
    


