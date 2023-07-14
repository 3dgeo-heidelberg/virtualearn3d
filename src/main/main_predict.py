# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
from src.main.main_train import MainTrain
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.model_io import ModelIO
from src.inout.point_cloud_io import PointCloudIO
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class MainPredict:
    """
    :author: Alberto M. Esmoris Pena

    Class handling the entry point for predictive tasks
    """
    # ---  MAIN METHOD   --- #
    # ---------------------- #
    @staticmethod
    def main(spec):
        """
        Entry point logic for predictive tasks.

        :param spec: Key-word specification
        """
        LOGGING.LOGGER.info('Starting prediction ...')
        start = time.perf_counter()
        pcloud = PointCloudFactoryFacade.make_from_file(
            MainTrain.extract_input_path(spec)
        )
        model = MainPredict.load_model(spec)
        preds = model.predict(pcloud)
        MainPredict.export_predictions(spec, pcloud, preds)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Predictions computed in {end-start:.3f} seconds.'
        )

    # ---  STATIC UTILS  --- #
    # ---------------------- #
    @staticmethod
    def load_model(spec):
        """
        Load the model from the specification of a predictive task.

        :param spec: Key-word specification
        :return: The loaded model.
        :rtype: :class:`.Model`
        """
        model_path = spec.get('in_model', None)
        if model_path is None:
            raise ValueError(
                'Computing predictions requires a model. None was given.'
            )
        return ModelIO.read(model_path)

    @staticmethod
    def export_predictions(spec, pcloud, preds):
        """
        Export the predictions as requested. There are two potential exports.

        1) Exporting the input point cloud with a new attribute named
            "prediction".

        2) Exporting the predictions as an ASCII CSV where each row (line)
            represents a point.

        :param spec: The specification on how to export the predictions.
        :type spec: dict
        :param pcloud: The predicted point cloud.
        :type pcloud: :class:`.PointCloud`
        :param preds: The predictions themselves.
        :type preds: :class:`np.ndarray`
        """
        # Point cloud export
        out_pcloud_path = spec.get('out_pcloud', None)
        pcloud = pcloud.add_features(
            ['prediction'], preds.reshape((-1, 1)), ftypes=preds.dtype
        )
        PointCloudIO.write(pcloud, out_pcloud_path)
        # CSV export
        out_csv_path = spec.get('out_preds', None)
        fmt = '%d' if np.issubdtype(preds.dtype, np.integer) else '%.8f'
        np.savetxt(out_csv_path, preds, fmt=fmt)
