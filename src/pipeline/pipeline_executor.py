# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.pcloud.point_cloud import PointCloud
from src.mining.miner import Miner
from src.utils.imput.imputer import Imputer
from src.utils.ftransf.feature_transformer import FeatureTransformer, \
    FeatureTransformerException
from src.utils.ctransf.class_transformer import ClassTransformer
from src.model.model_op import ModelOp
from src.eval.evaluator import Evaluator
from src.inout.writer import Writer
from src.inout.model_writer import ModelWriter
from src.pipeline.predictive_pipeline import PredictivePipeline
from src.inout.predictive_pipeline_writer import PredictivePipelineWriter
from src.inout.predictions_writer import PredictionsWriter
from src.inout.classified_pcloud_writer import ClassifiedPcloudWriter
import src.main.main_logger as LOGGING
import time


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
    :ivar pre_fnames: Cached feature names before preprocessing. Can be used
        to merge consecutive miners.
    :vartype pre_fnames: list
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
        self.pre_fnames = None
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

        See :meth:`pipeline_executor.PipelineExecutor.pre_process`,
        :meth:`pipeline_executor.PipelineExecutor.process`, and
        :meth:`pipeline_executor.PipelineExecutor.post_process`.

        :param state: The pipeline's state. See :class:`.PipelineState`.
        :param comp: The component to be executed.
        :param comp_id: The identifier of the component. Typically, it should
            be possible to use it to retrieve the component from comps.
        :param comps: The components composing the pipeline.
        :return: Nothing.
        """
        self.pre_process(state, comp, comp_id, comps)
        self.process(state, comp, comp_id, comps)
        self.post_process(state, comp, comp_id, comps)

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
            if isinstance(in_pcloud, PointCloud):
                state.update_pcloud(None, in_pcloud)
            elif isinstance(in_pcloud, str):
                state.update_pcloud(
                    None, PointCloudFactoryFacade.make_from_file(in_pcloud)
                )
            else:
                raise PipelineExecutorException(
                    'PipelineExecutor.load_input received an unexpected '
                    f'type for input point cloud: \"{type(in_pcloud)}\".'
                )
        # TODO Rethink : Handle model loading (if any) ?

    def pre_process(self, state, comp, comp_id, comps):
        """
        Handles the operations before the execution of the main logic, i.e.,
        before running the logic of the current component.

        See :meth:`pipeline_executor.PipelineExecutor.__call__`.
        """
        # Fill fnames automatically, if requested
        fnames_comp = comp  # First, extract component with fnames
        if isinstance(comp, ModelOp):
            fnames_comp = comp.model
        # Then, extract fnames
        fnames = getattr(fnames_comp, 'fnames', None)
        if fnames is not None and len(fnames) > 0:  # If feat. names are given
            if fnames[0] == "AUTO":  # If AUTO is requested
                fnames_comp.fnames = state.fnames  # Take from state
            else:  # Otherwise
                self.pre_fnames = state.fnames  # Cache fnames before update
                state.fnames = fnames_comp.fnames  # Set the state
        # Handle lazy preparation
        if hasattr(comp, 'lazy_prepare'):
            comp.lazy_prepare(state)

    def process(self, state, comp, comp_id, comps):
        """
        Handles the execution of the main logic, i.e., running the current
        component and updating the pipeline state consequently.

        See :meth:`pipeline_executor.PipelineExecutor.__call__`.
        """
        # Execute component
        if isinstance(comp, Miner):  # Handle miner
            LOGGING.LOGGER.info(
                f'Running {comp.__class__.__name__} data miner...'
            )
            start = time.perf_counter()
            state.update(comp, new_pcloud=comp.mine(state.pcloud))
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'{comp.__class__.__name__} data miner executed in '
                f'{end-start:.3f} seconds.'
            )
        elif isinstance(comp, Imputer):  # Handle imputer
            state.update(comp, new_pcloud=comp.impute_pcloud(state.pcloud))
        elif isinstance(comp, FeatureTransformer):  # Handle feat. transf.
            # Compute component logic and update pipeline state
            state.update(
                comp,
                new_pcloud=comp.transform_pcloud(
                    state.pcloud, out_prefix=self.out_prefix
                )
            )
        elif isinstance(comp, ClassTransformer):  # Handle class transformer
            # Compute component logic and update pipeline state
            state.update(
                comp,
                new_pcloud=comp.transform_pcloud(
                    state.pcloud, out_prefix=self.out_prefix
                )
            )
        elif isinstance(comp, ModelOp):
            if comp.op == ModelOp.OP.TRAIN:
                # Handle train
                state.update(
                    comp,
                    new_model=comp(state.pcloud, out_prefix=self.out_prefix)
                )
            elif comp.op == ModelOp.OP.PREDICT:
                # Handle predict
                state.update(
                    comp,
                    new_model=comp.model,
                    new_preds=comp(state.pcloud, out_prefix=self.out_prefix)
                )
                state.pcloud.add_features(
                    ['prediction'],
                    state.preds.reshape((-1, 1)),
                    ftypes=state.preds.dtype
                )
            else:
                raise PipelineExecutorException(
                    'PipelineExecutor received an unexpected model operation: '
                    f'{comp.op}'
                )
        elif isinstance(comp, PredictivePipeline):
            # Handle predict
            comp.pps.external_state = state
            new_preds = comp.predict(
                state.pcloud, out_prefix=self.out_prefix
            )
            if new_preds is not None:
                state.update(
                    comp,
                    new_preds=new_preds,
                    new_model=comp.get_first_model()
                )
            elif not comp.pps.ignore_predictions:
                raise PipelineExecutorException(
                    'PipelineExecutor called predictive pipeline without '
                    'ignore predictions flag but no predictions were computed.'
                )
        elif isinstance(comp, Evaluator):
            # Handle evaluation
            if hasattr(comp, 'eval_args_from_state'):
                comp(
                    **comp.eval_args_from_state(state),
                    y=state.pcloud.get_classes_vector(),
                    out_prefix=self.out_prefix
                )
            else:
                # Make sure state has predictions before evaluating
                if state.preds is None:
                    start = time.perf_counter()
                    state.preds = state.model.predict(state.pcloud)
                    end = time.perf_counter()
                    LOGGING.LOGGER.info(
                        'Missing predictions computed before evaluation in '
                        f'{end-start:.3f} seconds.'
                    )
                # Evaluate
                comp(
                    state.preds,
                    y=state.pcloud.get_classes_vector(),
                    out_prefix=self.out_prefix
                )
        elif isinstance(comp, Writer):  # Handle writer
            if comp.needs_prefix():
                if self.out_prefix is None:
                    raise PipelineExecutorException(
                        f"{self.maker.__class__.__name__} is running a case "
                        "with no path prefix for the output.\n"
                        "This requirement is a MUST."
                    )
            if isinstance(comp, ModelWriter):
                comp.write(state.model, prefix=self.out_prefix)
            elif isinstance(comp, PredictivePipelineWriter):
                comp.write(self.maker, prefix=self.out_prefix)
            elif isinstance(comp, PredictionsWriter):
                comp.write(
                    state.pcloud.get_features_matrix(['prediction']),
                    prefix=self.out_prefix
                )
            elif isinstance(comp, ClassifiedPcloudWriter):
                comp.write(state, prefix=self.out_prefix)
            else:
                comp.write(state.pcloud, prefix=self.out_prefix)

    def post_process(self, state, comp, comp_id, comps):
        """
        Handles the operations after the execution of the main logic, i.e.,
        after running the logic of the current component.

        See :meth:`pipeline_executor.PipelineExecutor.__call__`.
        """
        # Update the state's fnames considering posterior renames
        if isinstance(comp, FeatureTransformer):
            try:  # Try to obtain the transformed names without input fnames
                state.fnames = comp.get_names_of_transformed_features()
            except FeatureTransformerException as ftex:  # Try explicit input
                state.fnames = comp.get_names_of_transformed_features(
                    fnames=comp.fnames
                )
        # Merge feature names from different miners
        if isinstance(comp, Miner):
            if self.pre_fnames is not None:  # If previous fnames, merge them
                state.fnames = self.pre_fnames + state.fnames
