# ---   IMPORTS   --- #
# ------------------- #
from src.pipeline.pipeline import Pipeline, PipelineException
from src.pipeline.predictive_pipeline import PredictivePipeline
from src.pipeline.pps.pps_sequential import PpsSequential
from src.pipeline.state.simple_pipeline_state import SimplePipelineState
from src.pipeline.pipeline_executor import PipelineExecutor
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.pcloud.point_cloud_filter import PointCloudFilter
import src.main.main_logger as LOGGING
from src.main.main_mine import MainMine
from src.main.main_train import MainTrain
from src.main.main_eval import MainEval
from src.mining.miner import Miner
from src.utils.imput.imputer import Imputer
from src.utils.ftransf.feature_transformer import FeatureTransformer
from src.utils.ctransf.class_transformer import ClassTransformer
from src.utils.imputer_utils import ImputerUtils
from src.utils.ftransf_utils import FtransfUtils
from src.utils.ctransf_utils import CtransfUtils
from src.model.model_op import ModelOp
from src.inout.writer import Writer
from src.inout.writer_utils import WriterUtils
from src.inout.predictive_pipeline_writer import PredictivePipelineWriter
from src.inout.pipeline_io import PipelineIO
from src.inout.model_io import ModelIO
import numpy as np
import time
import copy


# ---   CLASS   --- #
# ----------------- #
class SequentialPipeline(Pipeline):
    """
    :author: Alberto M. Esmoris Pena

    Sequential pipeline (no loops, no recursion).
    See :class:`Pipeline`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of SequentialPipeline.
        A sequential pipeline execute the different components in the order
        they are given.
        See parent :class:`SequentialPipeline`

        :param kwargs: The attributes for the SequentialPipeline
        :ivar sequence: The sequence of components defining the
            SequentialPipeline.
        :vartype sequence: list
        """
        # Call parent init
        super().__init__(**kwargs)
        # Initialize state
        self.state = SimplePipelineState()
        # Pipeline as sequential list of components
        self.sequence = []
        seq_pipeline = kwargs.get('sequential_pipeline', None)
        if seq_pipeline is None:
            raise PipelineException(
                'Critical error initializing SequentialPipeline.\n'
                'There is no sequential_pipeline specification.\n'
                'This MUST not happen. Please, contact the developers.'
            )
        for comp in seq_pipeline:
            # Handle each potential component type
            if comp.get("miner", None) is not None:  # Handle miner
                miner_class = MainMine.extract_miner_class(comp)
                miner = miner_class(**miner_class.extract_miner_args(comp))
                self.sequence.append(miner)
            if comp.get("imputer", None) is not None:  # Handle imputer
                imputer_class = ImputerUtils.extract_imputer_class(comp)
                imputer = imputer_class(
                    **imputer_class.extract_imputer_args(comp)
                )
                self.sequence.append(imputer)
            if comp.get("feature_transformer", None) is not None:  # Hdl. ftr.
                ftransf_class = FtransfUtils.extract_ftransf_class(comp)
                ftransf = ftransf_class(
                    **ftransf_class.extract_ftransf_args(comp)
                )
                self.sequence.append(ftransf)
            if comp.get('class_transformer', None) is not None:  # Hdl. ctr.
                ctransf_class = CtransfUtils.extract_ctransf_class(comp)
                ctransf = ctransf_class(
                    **ctransf_class.extract_ctransf_args(comp)
                )
                self.sequence.append(ctransf)
            if comp.get("train", None) is not None:  # Handle train
                model_class = MainTrain.extract_model_class(comp)
                pretrained = comp.get('pretrained_model', None)
                if pretrained is not None:
                    model = MainTrain.extract_pretrained_model(
                        comp, model_class
                    )
                    model.overwrite_pretrained_model(comp)
                else:
                    model = model_class(**model_class.extract_model_args(comp))
                self.sequence.append(ModelOp(model, ModelOp.OP.TRAIN))
            if comp.get("predict", None) is not None:  # Handle predict
                if comp['predict'].lower() == 'predictivepipeline':
                    self.sequence.append(PipelineIO.read_predictive_pipeline(
                        comp['model_path']
                    ))
                elif comp['predict'].lower() == 'modelloader':
                    model = ModelIO.read(comp['model_path'])
                    self.sequence.append(ModelOp(model, ModelOp.OP.PREDICT))
                else:
                    raise PipelineException(
                        'SequentialPipeline does not support requested '
                        f'predict specification: "{comp["predict"]}"'
                    )
            if comp.get("eval", None) is not None:  # Handle eval
                eval_class = MainEval.extract_eval_class(comp)
                eval = eval_class(**eval_class.extract_eval_args(comp))
                self.sequence.append(eval)
            # Handle writer as component
            if comp.get("writer", None) is not None:  # Handle writer
                writer_class = WriterUtils.extract_writer_class(comp)
                writer = writer_class(**writer_class.extract_writer_args(comp))
                self.sequence.append(writer)
            # Handle potential writer for current non-writer component
            elif comp.get('out_pcloud', None) is not None:
                self.sequence.append(Writer(comp['out_pcloud']))

    # ---  RUN PIPELINE  --- #
    # ---------------------- #
    def run(self):
        """
        Run the sequential pipeline.

        See :meth:`sequential_pipeline.SequentialPipeline.run_for_in_pcloud`
        and
        :meth:`sequential_pipeline.SequentialPipeline.run_for_in_pcloud_concat`
        .

        :return: Nothing.
        """
        # Check that only a single input specification is given
        if self.in_pcloud is not None and self.in_pcloud_concat is not None:
            raise PipelineException(
                'SequentialPipeline found in_pcloud and in_pcloud_concat '
                'at the same time. This is ambiguous and is not supported. '
                'Please, choose a single input specification.'
            )
        # Handle in_pcloud input specification
        if self.in_pcloud is not None:
            self.run_for_in_pcloud()
        elif self.in_pcloud_concat is not None:
            self.run_for_in_pcloud_concat()
        else:
            raise PipelineException(
                'SequentialPipeline did NOT receive any input specification.'
            )

    def run_for_in_pcloud(self):
        """
        Run the sequential pipeline considering in_pcloud as the input
        specification.

        See :meth:`sequential_pipeline.SequentialPipeline.run`.

        :return: Nothing.
        """
        # List of input point clouds (even if just one is given)
        in_pclouds = self.in_pcloud
        if isinstance(in_pclouds, str):
            in_pclouds = [in_pclouds]
        # For each input point cloud, run the pipeline
        for i, in_pcloud in enumerate(in_pclouds):
            # Handle path to output point cloud, if any
            out_pcloud = None
            if self.out_pcloud is not None:
                out_pcloud = self.out_pcloud[i]
            # Save original sequence
            sequence = copy.deepcopy(self.sequence)
            # Run the pipeline for the current case
            self.run_case(in_pcloud, out_pcloud=out_pcloud)
            # Restore original sequence for next cases
            self.sequence = sequence

    def run_for_in_pcloud_concat(self):
        """
        Run the sequential pipeline considering in_pcloud_concat as the input
        specification.

        See :meth:`sequential_pipeline.SequentialPipeline.run`.

        :return: Nothing.
        """
        X = []  # Coordinates of the concatenated point cloud
        F = []  # Features of the concatenated point cloud
        y = []  # Classification of the concatenated point cloud
        fnames = None  # Feature names of the concatenated point cloud
        header = None  # Header of the concatenated point cloud
        # Load the concatenated input point cloud
        for i, concat in enumerate(self.in_pcloud_concat):
            in_pcloud_i = concat['in_pcloud']
            conditions_i = concat.get('conditions', None)
            # Load and filter input point cloud i
            pcloud_i = PointCloudFactoryFacade.make_from_file(in_pcloud_i)
            pcloud_i = PointCloudFilter(conditions_i).filter(pcloud_i)
            if i == 0:  # Get feature names and header from first point cloud
                fnames = pcloud_i.get_features_names()
                header = pcloud_i.get_header()
            # Concatenate coordinates, features, and classes
            X.append(pcloud_i.get_coordinates_matrix())
            if len(fnames) > 0:
                F.append(pcloud_i.get_features_matrix(fnames))
            if pcloud_i.has_classes():
                y.append(pcloud_i.get_classes_vector())
        # Concatenations to arrays
        X = np.vstack(X)
        F = np.vstack(F)
        y = np.concatenate(y)
        # Build concatenated input point cloud
        in_pcloud = PointCloudFactoryFacade.make_from_arrays(
            X, F, y=y, header=header, fnames=fnames
        )
        # Get path to output point cloud
        out_pcloud = self.out_pcloud
        if isinstance(self.out_pcloud, (list, tuple)):
            if len(self.out_pcloud) > 1:
                raise PipelineException(
                    'SequentialPipeline with in_pcloud_concat input is not '
                    'compatible with more than one output path but '
                    f'{len(self.out_pcloud)} were given.'
                )
            out_pcloud = self.out_pcloud[0]
        # Run pipeline
        self.run_case(in_pcloud, out_pcloud=out_pcloud)

    def run_case(self, in_pcloud, out_pcloud=None):
        """
        Run the sequential pipeline for a particular input point cloud.

        :param in_pcloud: The input point cloud for this particular case.
        :param out_pcloud: Optionally, the output path or prefix.
        :return: Nothing.
        """
        if isinstance(in_pcloud, str):
            LOGGING.LOGGER.info(
                f'SequentialPipeline running for "{in_pcloud}" ...'
            )
        else:
            LOGGING.LOGGER.info(
                f'SequentialPipeline running for {in_pcloud.get_num_points()} '
                'points ...'
            )
        start = time.perf_counter()
        # Prepare case (iteration)
        self.state.prepare_iter()
        # Prepare executor
        executor = PipelineExecutor(self, out_prefix=out_pcloud)
        executor.load_input(self.state, in_pcloud=in_pcloud)
        # Run pipeline
        for i, comp in enumerate(self.sequence):
            executor(self.state, comp, i, self.sequence)
        # If the pipeline's out_pcloud is a full path, use it automatically
        if out_pcloud is not None and out_pcloud[-1] != "*":
            writer = Writer(out_pcloud)
            if not writer.needs_prefix():
                writer.write(self.state.pcloud)
        # Report execution time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'SequentialPipeline for "{in_pcloud}" '
            f'computed in {end-start:.3f} seconds.'
        )

    # ---  PIPELINE METHODS  --- #
    # -------------------------- #
    def to_predictive_pipeline(self, **kwargs):
        """
        See :class:`.Pipeline` and
        :meth:`.pipeline.Pipeline.to_predictive_pipeline`.
        """
        # Copy itself
        sp = copy.copy(self)
        # Reinitialize state
        LOGGING.LOGGER.debug(
            'Sequential pipeline state at the moment when predictive pipeline '
            'generated:\n'
            f'Point cloud: {sp.state.pcloud}\n'
            f'Model: {sp.state.model}\n'
            f'Feature names: {sp.state.fnames}'
        )
        sp.state = SimplePipelineState()
        LOGGING.LOGGER.debug(
            'State of the generated predictive pipeline:\n'
            f'Point cloud: {sp.state.pcloud}\n'
            f'Model: {sp.state.model}\n'
            f'Feature names: {sp.state.fnames}'
        )
        # But copy only the desired components
        sp.sequence = []
        for comp in self.sequence:
            if isinstance(comp, ModelOp):
                sp.sequence.append(ModelOp(comp.model, ModelOp.OP.PREDICT))
            if isinstance(comp, Writer) and \
                    kwargs.get('include_writer', False) and \
                    not isinstance(comp, PredictivePipelineWriter):
                sp.sequence.append(comp)
            if isinstance(comp, Imputer) and \
                    kwargs.get('include_imputer', True):
                sp.sequence.append(comp)
            if isinstance(comp, FeatureTransformer) and \
                    kwargs.get('include_feature_transformer', True):
                sp.sequence.append(comp)
            if isinstance(comp, ClassTransformer) and \
                    kwargs.get('include_class_transformer', True):
                sp.sequence.append(comp)
            if isinstance(comp, Miner) and \
                    kwargs.get('include_miner', True):
                sp.sequence.append(comp)
        LOGGING.LOGGER.debug(
            'Sequence of original sequential pipeline has {m} components.\n'
            'Sequence of predictive sequential pipeline has {n} components.'
            .format(
                m=len(self.sequence),
                n=len(sp.sequence)
            )
        )
        # Build the predictive pipeline
        pp = PredictivePipeline(sp, PpsSequential())
        # Clear paths of predictive pipeline
        pp.in_pcloud = None
        pp.out_pcloud = None
        # Return the predictive pipeline
        return pp

    def is_using_deep_learning(self):
        """
        A sequential pipeline is said to use deep learning if it contains at
        least one ModelOp which is based on a deep learning model.

        See :meth:`pipeline.Pipeline.is_using_deep_learning`.
        """
        for comp in self.sequence:
            if(
                isinstance(comp, ModelOp) and
                comp.model.is_deep_learning_model()
            ):
                return True
        return False

    def write_deep_learning_model(self, path):
        """
        Write the deep learning models contained in the sequential pipeline.

        See :meth:`pipeline.Pipeline.write_deep_learning_model`.
        """
        # Write deep learning models
        num_model = 1
        for comp in self.sequence:
            if(
                isinstance(comp, ModelOp) and
                comp.model.is_deep_learning_model()
            ):
                _path = path if num_model == 1 else f'{num_model}_{path}'
                ModelIO.write(comp.model, _path)
                LOGGING.LOGGER.debug(
                    f'Deep learning model written to "{_path}"'
                )
                num_model += 1

