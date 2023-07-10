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
        :param comp_id: The identified of the component. Typically, it should
            be possible to use it to retrieve the component from comps.
        :param comps: The components composing the pipeline.
        :return: Nothing.
        """
        if isinstance(comp, Miner):  # Handle miner
            state.update(
                comp, new_pcloud=comp.mine(state.pcloud)
            )
        elif isinstance(comp, Imputer):  # Handle imputer
            state.update(
                comp, new_pcloud=comp.impute_pcloud(state.pcloud)
            )
        elif isinstance(comp, FeatureTransformer):  # Handle feat. transf.
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
