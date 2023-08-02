# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.dlrun.grid_subsampling_pre_processor import \
    GridSubsamplingPreProcessor
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import scipy.stats
import numpy as np
import joblib
import time

# ---   CLASS   --- #
# ----------------- #
class FurthestPointSubsamplingPreProcessor:
    """
    # TODO Rethink : Doc
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization/instantiation of a Furthest Point Subsampling
        pre-processor.

        :param kwargs: The key-word arguments for the
            FurthestPointSubsamplingPreProcessor.
        """
        # Assign attributes
        self.num_points = kwargs.get('num_points', 8000)
        self.num_encoding_neighbors = kwargs.get('num_encoding_neighbors', 3)
        self.fast = kwargs.get('fast', False)
        self.neighborhood_spec = kwargs.get('neighborhood', None)
        if self.neighborhood_spec is None:
            raise DeepLearningException(
                'The FurthestPointSubsamplingPreProcessor did not receive '
                'any neighborhood specification.'
            )
        self.nthreads = kwargs.get('nthreads', 1)
        # Initialize last call cache
        self.last_call_receptive_fields = None
        self.last_call_neighborhoods = None

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        # TODO Rethink : Doc
        """
        # Extract inputs
        start = time.perf_counter()
        X, y = inputs['X'], inputs.get('y', None)
        # Extract neighborhoods
        # TODO Rethink : Consider 2D support points for unbounded cylinders
        sup_X = GridSubsamplingPreProcessor.build_support_points(
            X,
            self.neighborhood_spec['separation_factor'],
            self.neighborhood_spec['radius']
        )
        kdt = KDT(X)
        kdt_sup = KDT(sup_X)
        I = kdt_sup.query_ball_tree(kdt, self.neighborhood_spec['radius'])
        # Remove empty neighborhoods and corresponding support points
        I, sup_X = FurthestPointSubsamplingPreProcessor\
            .clean_support_neighborhoods(
                sup_X, I, self.num_points
            )
        self.last_call_neighborhoods = I
        # Prepare receptive field
        self.last_call_receptive_fields = [
            ReceptiveFieldFPS(
                num_points=self.num_points,
                num_encoding_neighbors=self.num_encoding_neighbors,
                fast=self.fast
            )
            for Ii in I
        ]
        self.last_call_receptive_fields = joblib.Parallel(
            n_jobs=self.nthreads
        )(
            joblib.delayed(
                self.last_call_receptive_fields[i].fit
            )(
                X[Ii], sup_X[i]
            )
            for i, Ii in enumerate(I)
        )
        # Neighborhoods ready to be fed into the neural network
        Xout = np.array([
            self.last_call_receptive_fields[i].centroids_from_points(None)
            for i in range(len(I))
        ])
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'The furthest point subsampling pre processor generated '
            f'{Xout.shape[0]} receptive fields.'
        )
        # TODO Rethink : Duplicated code wrt GS subsampling
        if y is not None:
            yout = np.array(joblib.Parallel(n_jobs=self.nthreads)(
                joblib.delayed(
                    self.last_call_receptive_fields[i].reduce_values
                )(
                    Xout[i],
                    y[Ii],
                    reduce_f=lambda x: scipy.stats.mode(x)[0][0]
                ) for i, Ii in enumerate(I)
            ))
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'The furthest point subsampling pre processor pre-processed '
                f'{X.shape[0]} points for training in {end-start:.3f} seconds.'
            )
            return Xout, yout
        LOGGING.LOGGER.info(
            'The furthest point subsampling pre processor pre-processed '
            f'{X.shape[0]} points for predictions in {end-start:.3f} seconds.'
        )
        return Xout

    # ---   POINT-NET METHODS   --- #
    # ----------------------------- #
    def get_num_input_points(self):
        """
        See
        :meth:`point_net_pre_processor.PointNetPreProcessor.get_num_input_points`
        .
        """
        return self.num_points

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def clean_support_neighborhoods(sup_X, I, num_points):
        """
        # TODO Rethink : Doc
        """
        # TODO Rethink : Is this logic okay?
        # Remove neighborhoods with less than num_points neighbors
        non_empty_mask = [len(Ii) >= num_points for Ii in I]
        I = [Ii for i, Ii in enumerate(I) if non_empty_mask[i]]
        sup_X = sup_X[non_empty_mask]
        return I, sup_X
