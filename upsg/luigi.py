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

    req_tasks = {key: context[conn.other.node.uid] for
                 key, conn in get_inputs().iteritems()}
    def requires(self):
        return req_tasks

    # TODO nonlocal targets
    out_files = [luigi.LocalTarget('{}.upsg'.format(
                    conn.edge.uid)) for conn  in get_outputs.itervalues()]
    def output(self):
        return out_files

    def run(self):




