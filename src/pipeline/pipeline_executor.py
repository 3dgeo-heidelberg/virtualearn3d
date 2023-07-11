# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.mining.miner import Miner
from src.utils.imput.imputer import Imputer
from src.utils.ftransf.feature_transformer import FeatureTransformer
from src.model.model_op import ModelOp
from src.inout.writer import Writer
from src.inout.model_writer import ModelWriter


# ---   EXCEPTIONS   --- #
# ---------------------- #
class PipelineExecutorException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to the execution of pipelines.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class PipelineExecutor:
    """
    :author: Alberto M. Esmoris Pena

    Class to handle the execution of components in the context of a pipeline.

    :ivar maker: The pipeline that instantiated the executor.
    :vartype maker: :class:`.Pipeline`
    :ivar out_prefix: Optional attribute (can be None) that specifies the
        output prefix for any component that needs to append it to its output
        paths.
    :vartype out_prefix: str
    :ivar recent_fnames_id: The identifier of the most recent known component
        which fnames have been automatically considered. It is initialized to
        zero because it governs the stop condition for backward searches.
    :vartype recent_fnames_idx: int
    :ivar recent_fnames: The most recent known feature names. It is initialized
        to None.
    :vartype recent_fnames: list
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, maker, **kwargs):
        """
        Handle the root-level (most basic) initialization of any pipeline
        executor.

        :param maker: The pipeline that instantiated the executor.
        :type maker: :class:`.Pipeline`
        :param kwargs: The attributes for the PipelineExecutor
        """
        # Fundamental attributes for any pipeline executor
        self.maker = maker
        self.out_prefix = kwargs.get('out_prefix', None)
        self.recent_fnames_id = 0
        self.recent_fnames = None
        # Validate
        if self.maker is None:
            raise PipelineExecutorException(
                'A PipelineExecutor always requires an associated pipeline.'
            )

    # ---  PIPELINE EXECUTOR METHODS  --- #
    # ----------------------------------- #
    def __call__(self, state, comp, comp_id, comps):
        """
        Execute the component of the pipeline associated to the given
        identifier.

        By default, comp_id is expected to be an integer and comps a list such
        that comps[comp_id] = comp.

        :param state: The pipeline's state. See :class:`.PipelineState`.
        :param comp: The component to be executed.
        :param comp_id: The identifier of the component. Typically, it should
            be possible to use it to retrieve the component from comps.
        :param comps: The components composing the pipeline.
        :return: Nothing.
        """
        # Fill fnames automatically, if requested
        self.autofill_fnames(comp, comp_id, comps)
        # Execute component
        if isinstance(comp, Miner):  # Handle miner
            state.update(
                comp, new_pcloud=comp.mine(state.pcloud)
            )
        elif isinstance(comp, Imputer):  # Handle imputer
            state.update(
                comp, new_pcloud=comp.impute_pcloud(state.pcloud)
            )
        elif isinstance(comp, FeatureTransformer):  # Handle feat. transf.
            # Compute component logic and update pipeline state
            state.update(
                comp,
                new_pcloud=comp.transform_pcloud(
                    state.pcloud, out_prefix=self.out_prefix
                )
            )
        elif isinstance(comp, ModelOp) and comp.op == ModelOp.OP.TRAIN:
            # Handle train
            state.update(
                comp,
                new_model=comp(state.pcloud, out_prefix=self.out_prefix)
            )
        # TODO Rethink : Add elif for train, predict, and eval too
        elif isinstance(comp, Writer):  # Handle writer
            if comp.needs_prefix():
                if self.out_prefix is None:
                    raise PipelineExecutorException(
                        f"{self.maker.__name__} is running a case with no "
                        "path prefix for the output.\n"
                        "This requirement is a MUST."
                    )
                if isinstance(comp, ModelWriter):
                    comp.write(state.model, prefix=self.out_prefix)
                else:
                    comp.write(state.pcloud, prefix=self.out_prefix)
            else:
                comp.write(state.pcloud)

    def load_input(self, state, **kwargs):
        """
        Load the input point cloud in the pipeline's state.

        :param state: The pipeline's state.
        :type state: :class:`.PipelineState`
        :param kwargs: The key-word arguments. They can be used to specify
            the path to the input point cloud through the "in_pcloud" key.
        :return: Nothing but the state is updated.
        """
        # Load input point cloud
        in_pcloud = kwargs.get('in_pcloud', None)
        if in_pcloud is not None:
            state.update_pcloud(
                None, PointCloudFactoryFacade.make_from_file(in_pcloud)
            )
        # TODO Rethink : Handle model loading (if any)

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def autofill_fnames(self, comp, comp_id, comps):
        """
        Fill the feature names of the component considering the previous
        components. It is assumed that comps[i] for i=0,...,comp_id is
        supported.

        :param comp: The component to be executed.
        :param comp_id: The identifier of the component. Typically, it should
            be possible to use it to retrieve the component from comps.
        :param comps: The components composing the pipeline.
        :return: Nothing.
        """
        # Extract component with fnames
        fnames_comp = comp
        if isinstance(comp, ModelOp):
            fnames_comp = comp.model
        # Extract fnames
        fnames = getattr(fnames_comp, 'fnames', None)
        if fnames is not None and fnames[0] == "AUTO":  # If AUTO is requested
            recent_fnames = None  # Find the most recent known fnames
            stop_id = self.recent_fnames_id-1
            for i in range(comp_id-1, stop_id, -1):  # Backward search
                # Extract the component associated to feature names
                comps_i = comps[i]
                if isinstance(comps_i, ModelOp):
                    comps_i = comps_i.model
                # Get frenames and, if they are not available, try fnames
                recent_fnames = getattr(comps_i, 'frenames', None)
                if recent_fnames is None:
                    recent_fnames = getattr(comps_i, 'fnames', None)
                if recent_fnames is not None:
                    self.recent_fnames_id = comp_id+1
                    self.recent_fnames = recent_fnames
                    break
            # Make the update on feature names effective
            fnames_comp.fnames = self.recent_fnames
            # Update the recent_fnames of the executor for posterior renames
            if isinstance(fnames_comp, FeatureTransformer):
                self.recent_fnames = \
                    fnames_comp.get_names_of_transformed_features()
