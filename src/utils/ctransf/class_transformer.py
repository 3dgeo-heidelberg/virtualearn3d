# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
import numpy as np


# ---   EXCEPTIONS   --- #
# ---------------------- #
class ClassTransformerException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to class transformation components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class ClassTransformer:
    """
    :author: Alberto M. Esmoris Pena

    Class (code) for class (classification) transformation operations.

    :ivar num_classes: The number of input classes
    :vartype num_classes: int
    :ivar input_class_names: The list of names corresponding to input classes.
        For example [0] is the name of the first class, i.e., that represented
        with index 0.
    :vartype input_class_names: list of str
    :ivar report_path: The path to write the report file reporting the behavior
        of the class transformer.
    :vartype report_path: str
    :ivar on_predictions: Flag controlling whether the transformation must
        be applied to classification or prediction.
    :vartype on_predictions: bool
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ctransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a ClassTransformer
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a ClassTransformer.
        """
        # Initialize
        kwargs = {
            'num_classes': spec.get('num_classes', None),
            'input_class_names': spec.get('input_class_names', None),
            'report_path': spec.get('report_path', None),
            'on_predictions': spec.get('on_predictions', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a ClassTransformer.

        :param kwargs: The attributes for the ClassTransformer.
        """
        # Fundamental initialization of any class transformer
        self.num_classes = kwargs.get('num_classes', None)
        self.input_class_names = kwargs.get('input_class_names', None)
        self.report_path = kwargs.get('report_path', None)
        self.on_predictions = kwargs.get('on_predictions', None)
        # Automatically generate input class names from num classes
        if self.num_classes is not None and self.input_class_names is None:
            self.input_class_names = [
                str(f'C{i}') for i in range(self.num_classes)
            ]
        # Update num classes from input class names
        elif self.num_classes is None and self.input_class_names is not None:
            self.num_classes = len(self.input_class_names)
        # Exception : No num classes nor input class names given
        elif self.num_classes is None and self.input_class_names is None:
            raise ClassTransformerException(
                'ClassTransformer received no number of classes and no '
                'input class names.'
            )
        # Exception : Num classes different from length of input class names
        elif self.num_classes != len(self.input_class_names):
            raise ClassTransformerException(
                'ClassTransformer received a different number of classes than '
                'input class names.'
            )

    # ---  CLASS TRANSFORM METHODS  --- #
    # --------------------------------- #
    @abstractmethod
    def transform(self, y, out_prefix=None):
        """
        The fundamental transformation logic defining the class transformer.

        :param y: The vector of classes.
        :type y: :class:`np.ndarray`
        :param out_prefix: The output prefix (OPTIONAL). It might be used by
            a report to particularize the output path.
        :type out_prefix: str
        :return: The transformed vector of classes.
        :rtype: :class:`np.ndarray`
        """
        pass

    def transform_pcloud(self, pcloud, out_prefix=None):
        """
        Apply the transform method to a point cloud.

        See :meth:`class_transformer.ClassTransformer.transform`

        :param pcloud: The point cloud to be transformed.
        :type pcloud: :class:`.PointCloud`
        :param out_prefix: The output prefix (OPTIONAL). It might be used
            by a report to particularize the output path.
        :type out_prefix: str
        :return: A new point cloud that is the transformed version of the
            input point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Check point cloud has classes or predictions (as requested)
        if not self.on_predictions and not pcloud.has_classes():
            raise ClassTransformerException(
                'Class transformer cannot transform point cloud classes '
                'because point cloud has no "classification" attribute.'
            )
        if self.on_predictions and not pcloud.has_predictions():
            raise ClassTransformerException(
                'Class transformer cannot transform point cloud classes '
                'because point cloud has no "prediction" attribute.'
            )
        # Transform classes
        y = pcloud.get_predictions_vector() if self.on_predictions else \
            pcloud.get_classes_vector()
        y = self.transform(y, out_prefix=out_prefix)
        # Return new point cloud
        if self.on_predictions:  # With transformed "prediction"
            fnames = pcloud.get_features_names()
            prediction_idx = np.flatnonzero(np.array(fnames) == "prediction")
            if len(prediction_idx) > 0:
                F = np.delete(
                    pcloud.get_features_matrix(),
                    prediction_idx,
                    axis=1
                )
            else:
                F = pcloud.get_features_matrix()
            return PointCloudFactoryFacade.make_from_arrays(
                pcloud.get_coordinates_matrix(),
                np.hstack([F, y.reshape(-1, 1)]),
                y=pcloud.get_classes_vector(),
                fnames=fnames
            )
        else:  # With transformed "classification"
            return PointCloudFactoryFacade.make_from_arrays(
                pcloud.get_coordinates_matrix(),
                pcloud.get_features_matrix(),
                y=y,
                header=self.build_new_las_header(pcloud),
                fnames=pcloud.get_features_names()
            )

    def report(self, report, out_prefix=None):
        """
        Handle the way a report is reported. First, it will be reported using
        the logging system. Then, it will be written to a file if the
        transformer has a not None report_path.


        :param report: The report to be reported.
        :param out_prefix: The output prefix in case the output path must be
            expanded.
        :return: Nothing.
        """
        LOGGING.LOGGER.info(report)
        if self.report_path is not None:
            report.to_file(self.report_path, out_prefix=out_prefix)

    def build_new_las_header(self, pcloud):
        """
        Build the LAS header for the output point cloud.

        See :class:`.PointCloud` and
        :meth:`class_transformer.ClassTransformer.transform_pcloud`.

        :param pcloud: The input point cloud as reference to build the header
            for the new point cloud.
        :type pcloud: :class:`.PointCloud`
        :return: The LAS header for the output point cloud.
        """
        return pcloud.las.header
