class Pipeline:
    """Internal representation of a UPSG pipeline.

    Our structure is merely a graph of pipeline elements. Execution will
    be relegated to either Drake or some simplified, internal replacement.
    
    """

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
        pass

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
        pass

    def run(self):
        """Run the pipeline"""
