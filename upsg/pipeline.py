from collections import namedtuple

class PipelineException(Exception):
    pass

class Connection:
    def __init__(self, key, outgoing, node):
        self.__key = key
        self.__other = None
        self.__outgoing = outgoing
        self.__node = node
    def connect_to(self, other):
        """

        Parameters
        ----------
        other : __Connection

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
        return self.__node

class Node:
        
    def __init__(self, stage):
        self.__stage = stage
        self.__connections = {key : Connection(key, False, self) for key 
            in stage.input_keys}
        self.__connections.update({key : Connection(key, True, self) 
            for key in stage.output_keys})

    def __getitem__(self, key):
        return self.__connections[key]

    def __repr__(self):
        return 'Node({})'.format(self.__stage)

    def get_stage(self):
        return self.__stage

    def get_inputs(self):
        #TODO raise an error if all of the required inputs have not been
        # connected yet
        return {key: self.__connections[key] for key in self.__connections
             if self.__connections[key].other is not None and not
                self.__connections[key].outgoing}

    def get_outputs(self):
        return {key: self.__connections[key] for key in self.__connections
             if self.__connections[key].other is not None and
                self.__connections[key].outgoing}

class Pipeline:
    """Internal representation of a UPSG pipeline.

    Our structure is merely a graph of pipeline elements. Execution will
    be relegated to either Drake or some simplified, internal replacement.
    
    """


    def __init__(self):
        self.__nodes = []

    def add(self, stage):
        """Add a stage to the pipeline

        Parameters
        ----------
        stage: an instance of Stage to add to the pipeline.

        Returns
        -------
        A Node encapsulating the given stage       

        """
        node = Node(stage)
        self.__nodes.append(node)
        return node


    def __integrate(self, other):
        """Integrates another pipeline into this one and creates a virtual
        uid to access the sub-pipeline.

        Parameters
        ----------
        other : Pipeline

        Returns
        -------
        uid which can be used to connect nodes to sub-pipeline as if the 
            sub-pipeline were a single node.

        """
        return NotImplementedError()

    def run_debug(self, verbose = False):
        """Run the pipeline in the current Python process.

        This method of running the job runs everything in serial on a single 
        process. It is provided for debugging purposes for use with small jobs.
        For larger and more performant jobs, use the run method.
        """
        #TODO what should the user call rather than run?
        node_queue = [node for node in self.__nodes 
            if not node.get_outputs()] # start with the root nodes
        state = dict.fromkeys(self.__nodes, None)  
        while node_queue:
            node = node_queue.pop()
            if state[node] is not None: # already computed
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
            input_args = {input_key: state[other][other_key] 
                for input_key, other, other_key 
                in map(lambda k: (k, input_connections[k].other.node, 
                    input_connections[k].other.key), input_connections)}
            output_args = node.get_stage().run(node.get_outputs().keys(), 
                **input_args)
            map(lambda k: output_args[k].to_read_phase(), output_args)
            if verbose:
                print node
                for arg in input_args:
                    print '<-{}: {}'.format(arg, input_args[arg].to_np())
                for arg in output_args:
                    print '->{}: {}'.format(arg, output_args[arg].to_np())
            state[node] = output_args

    def run(self, **kwargs):
        """Run the pipeline"""
        #TODO a better method of scheduling/running than this
        self.run_debug(**kwargs)
