# ---   IMPORTS   --- #
# ------------------- #
from src.pipeline.pps.pipeline_predictive_strategy import \
    PipelinePredictiveStrategy, PipelinePredictiveStrategyException
from src.mining.miner import Miner
from src.model.model_op import ModelOp
from src.utils.imput.imputer import Imputer
from src.utils.ftransf.feature_transformer import FeatureTransformer
from src.utils.ctransf.class_transformer import ClassTransformer
from src.inout.writer import Writer
import src.main.main_logger as LOGGING
import time


# ---   CLASS   --- #
# ----------------- #
class PpsSequential(PipelinePredictiveStrategy):
    """
    :author: Alberto M. Esmoris Pena

    A predictive strategy for sequential pipelines.
    See :class:`.PipelinePredictiveStrategy` and :class:`.SequentialPipeline`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Handles the initialization of a predictive strategy for sequential
        pipelines.

        :param kwargs: The attributes for the pipeline's predictive strategy.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   PREDICTIVE PIPELINE METHODS   --- #
    # --------------------------------------- #
    def predict(self, pipeline, pcloud):
        """
        See :class:`.PipelinePredictiveStrategy` and
        :meth:`pipeline_predictive_strategy.PipelinePredictiveStrategy.predict`
        .
        """
        num_points = pcloud.get_num_points()
        LOGGING.LOGGER.info(
            f'Sequential predictive pipeline on {num_points} '
            'points ...'
        )
        start = time.perf_counter()
        preds = self._predict(pipeline, pcloud)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Sequential predictive pipeline on {num_points} '
            f'points computed in {end-start:.3f} seconds.'
        )
        return preds

    def _predict(self, pipeline, pcloud):
        """
        The logic of the predict method.
        See :meth:`pps_sequential.PpsSequential.predict`.
        """
        # Initialize
        preds = None
        # Run the wrapped pipeline
        for i, comp in enumerate(pipeline.sequence):
            if isinstance(comp, Miner):  # Handle miner
                LOGGING.LOGGER.info(
                    f'Running {comp.__class__.__name__} data miner...'
                )
                start = time.perf_counter()
                pcloud = comp.mine(pcloud)
                end = time.perf_counter()
                LOGGING.LOGGER.info(
                    f'{comp.__class__.__name__} data miner executed in '
                    f'{end-start:.3f} seconds.'
                )
            elif isinstance(comp, Writer):  # Handle writer
                if comp.needs_prefix():
                    if self.out_path is None:
                        raise PipelinePredictiveStrategyException(
                            'A Writer in the sequential predictive pipeline '
                            'needs an output prefix to write. '
                            'None was given.'
                        )
                    comp.write(pcloud, prefix=self.out_path)
                else:
                    comp.write(pcloud)
            elif isinstance(comp, ModelOp):  # Handle model
                if comp.op == ModelOp.OP.TRAIN:
                    raise PipelinePredictiveStrategyException(
                        'The sequential pipeline predictive strategy does not '
                        'support model training.'
                    )
                # Handle model prediction
                print('PPSSequential predicting ...')  # TODO Remove
                preds = comp(pcloud=pcloud, out_prefix=self.out_path)
                print('PPSSequential predicted!')  # TODO Remove
            elif isinstance(comp, Imputer):  # Handle imputer
                pcloud = comp.impute_pcloud(pcloud)
            elif isinstance(comp, FeatureTransformer):  # Handle feat. transf.
                pcloud = comp.transform_pcloud(
                    pcloud, out_prefix=self.out_path
                )
            elif isinstance(comp, ClassTransformer):
                pcloud = comp.transform_pcloud(
                    pcloud, out_prefix=self.out_path
                )
            else:  # Unexpected component
                raise PipelinePredictiveStrategyException(
                    'PipelinePredictiveStrategy received an unexpected '
                    f'component: "{comp.__class__.__name__}"'
                )
        # Validate
        if preds is None:
            raise PipelinePredictiveStrategyException(
                'The sequential pipeline predictive strategy failed to '
                'compute predictions.'
            )
        # Update given state point cloud, if any
        if self.external_state is not None:
            print('PPSSequential updating external state ...')  # TODO Remove
            self.external_state.pcloud.clear_data(proxy_release=True)
            self.external_state.pcloud = pcloud
            print('PPSSequential updated external state!')  # TODO Remove
        # Add predictions to point cloud
        print('PPSSequential adding features to point cloud ...')  # TODO Remove
        pcloud.add_features(
            ['prediction'], preds.reshape((-1, 1)), ftypes=preds.dtype
        )
        print('PPSSequential added features to point cloud!')  # TODO Remove
        # Return
        return preds
