from collections import namedtuple

class Pipeline:
    """Internal representation of a UPSG pipeline.

    Our structure is merely a graph of pipeline elements. Execution will
    be relegated to either Drake or some simplified, internal replacement.
    
    """

    class __Node:
        class __Connection:
            def __init__(self, my_key):
                self.__my_key = my_key
                self.__other = None
                self.__outgoing = False
            def connect_to(other):
                """

                Parameters
                ----------
                other : __Connection

                """
                self.__other = other
                self.__outgoing = True
                other.__other = self
                other.__outgoing = False
            def __gt__(self, other):
                self.connect_to(other)
            
        def __init__(self, stage):
            self.__stage = stage
            self.__in = dict.fromkeys(stage.input_keys, None)
            self.__out = dict.fromkeys(stage.output_keys, None)  

        def add_input(self, other, other_key, my_key):
            self.__in[my_key] = self.Connection(other, other_key)

        def add_output(self, other, other_key, my_key):
            self.__out[my_key] = self.Connection(other, other_key)

        def get_stage(self):
            return self.__stage

        def get_inputs(self):
            #TODO raise an error if all of the required inputs have not been
            # connected yet
            return {key: self.__in[key] for key in self.__in if self.__in[key] 
                is not None}

        def get_outputs(self):
            return {key: self.__out[key] for key in self.__out if 
                self.__out[key] is not None}

    def __init__(self):
        self.__next_node_uid = 0
        self.__nodes = {}

    def add(self, stage):
        """Add a stage to the pipeline

        In order to be utilized in the pipeline, a stage must be both added
        to the pipeline and connected to other stages using the connect method. 

        Parameters
        ----------
        stage: an instance of Stage to add to the pipeline.

        Returns
        -------
        A unique identifier for this stage to be used to connect this Stage
        to other Stages in the pipeline.
        """
        uid = self.__next_node_uid
        self.__nodes[uid] = self.__Node(stage)
        self.__next_node_uid += 1
        return uid

    def connect(self, from_stage_uid, from_stage_output_key, to_stage_uid,
        to_stage_input_key):
        """Connects the output from one Stage in the pipeline to the input of
        another Stage.

        Parameters
        ----------
        from_stage_uid: The return value of Pipeline.add corresponding to the 
        Stage that is generating some output.

        from_stage_output_key: Key corresponding to the element of the output
            stage's output that will be connected.

        to_stage_uid: The return value of Pipeline.add corresponding to the
            Stage that is receiving input.
    
        to_stage_input_key: The key for the argument of the input Stage that
            corresponds to the output Stage's output.

        """
        from_node = self.__nodes[from_stage_uid]
        to_node = self.__nodes[to_stage_uid]
        from_node.add_output(to_stage_uid, to_stage_input_key, 
            from_stage_output_key)
        to_node.add_input(from_stage_uid, from_stage_output_key, 
            to_stage_input_key)

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
        node_queue = [uid for uid in self.__nodes 
            if not self.__nodes[uid].get_outputs()] # start with the root nodes
        state = dict.fromkeys(self.__nodes.keys(), None)  
        while node_queue:
            uid = node_queue.pop()
            if state[uid] is not None: # already computed
                continue
            node = self.__nodes[uid]
            node_inputs = node.get_inputs()
            input_uids = frozenset([node_inputs[input_key].other for 
                input_key in node_inputs])
            unfinished_dependencies = [uid_in for uid_in in input_uids 
                if state[uid_in] is None] 
            if unfinished_dependencies:
                node_queue.append(uid)
                node_queue += unfinished_dependencies
                continue
            input_args = {input_key: state[other][other_key] 
                for input_key, other, other_key 
                in map(lambda k: (k, node_inputs[k].other, 
                    node_inputs[k].other_key), node_inputs)}
            output_args = node.get_stage().run(node.get_outputs().keys(), 
                **input_args)
            map(lambda k: output_args[k].to_read_phase(), output_args)
            if verbose:
                print 'uid: {}'.format(uid)
                for arg in output_args:
                    print "--{}: {}".format(arg, output_args[arg].to_np())
            state[uid] = output_args

    def run(self, **kwargs):
        """Run the pipeline"""
        #TODO a better method of scheduling/running than this
        self.run_debug(**kwargs)
