class Pipeline:
    """Internal representation of a UPSG pipeline.

    Our structure is merely a graph of pipeline elements. Execution will
    be relegated to either Drake or some simplified, internal replacement.
    
    """

    class __Node:
        def __init__(self, stage):
            self.__stage = stage
            self.__in = dict.fromkeys(stage.input_keys.keys(), None)
            self.__out = dict.fromkeys(stage.output_keys, None)  

        def add_input(other, other_key, my_key):
            self.__in[my_key] = (other, other_key)

        def add_output(other, other_key, my_key):
            self.__out[my_key] = (other, other_key)

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
        self.__nodes[uid] = Node(stage)
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
        from_node.add_output(to_node, to_stage_input_key, 
            from_stage_output_key)
        to_node.add_input(from_node, from_stage_output_key, 
            to_stage_input_key)

    def run(self):
        """Run the pipeline"""
        #TODO stub
        raise NotImplementedError()
