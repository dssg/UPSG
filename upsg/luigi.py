from collections import named

import luigi

# We have to traverse the graph, starting with nodes that require no inputs,
# and make them into tasks. Keep in mind that each task needs to be its own
# class, rather than its own instance of a class.
# or it could be a Parameter. It seems like generating a class is probably
# less of an abuse of the api

# In order to have a single target, we could have targets that correspond to
# multiple .upsg files

# Before we do anything, we have to put fileio in control of the pipeline
# rather than uobject. File names can correspond to edge names. We probably
# need some meta-information to figure out which input/output keys a given
# file goes to. UObject should probably open a file in memory and then actually
# write it at the last minute


def node_to_task(node, context):

    # we need to keep track of which node gives which output
    node_inputs = node.get_inputs()
    node_outputs = node.get_outputs()
    req_tasks = {key: context[conn.other.node] for
                 key, conn in node_inputs.iteritems()}
    def requires(self):
        return req_tasks

    # TODO nonlocal targets
    out_files = {key: luigi.LocalTarget('{}.upsg'.format(
                    conn.edge.uid)) for key, conn  in node_outputs.iteritems()}
    def output(self):
        return out_files

    others_output_keys = {in_key: node_inputs[in_key].other.node.key 
                          for in_key in node_inputs}
    def run(self):
        import pdb; pdb.set_trace()
        input_files = {in_key:
                       self.input()[in_key][others_output_keys[in_key]].open('r') 
                       for in_key in others_output_keys}
        input_args = {in_key:
                      UObject(
                          UObjectPhase.Read, 
                          input_files[in_key].read())}
        [input_files[in_key].close() for in_key in input_files]
        output_args = node.get_stage().run(node_outputs.keys(), **input_args)
        output_files = {out_key: self.output()[out_key].open('w') for 
                        out_key in node_outputs}
        [output_files[out_key].write(output_args[out_key].get_image()) for 
         out_key in output_files]
        [output_files[out_key].close() for out_key in output_files]

        task =  type(
            'Task_{}'.format(node.uid), 
            (luigi.Task,), 
            {'requires': requires, 'output': output, 'run': run})
        context[node] = task
        return task

    #TODO unfilled dependencies

def run_luigi(nodes):
    node_queue = [node for node in self.__nodes
                  if not node.get_outputs()]  # start with the leaves
    context = dict.fromkeys(self.__nodes, None)
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
        node_to_task(node, context)
    luigi.run()


