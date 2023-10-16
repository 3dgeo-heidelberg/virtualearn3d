# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ctransf.class_transformer import ClassTransformer, \
    ClassTransformerException
from src.report.class_reduction_report import ClassReductionReport
from src.plot.class_reduction_plot import ClassReductionPlot
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class ClassReducer(ClassTransformer):
    r"""
    :author: Alberto M. Esmoris Pena

    Class to reduce a given set of input classes into another set of output
    classes such that each class of the output set corresponds at least 1
    (but potentially more) classes of the input set.

    More formally, this class represents a map from m input classes to n output
    classes where :math:`m \geq n`.

    See :class:`.ClassTransformer`.

    :ivar output_class_names: The list of names for the output (transformed)
        classes. For example [0] is the name of the first output class, i.e.,
        that represented with index 0.
    :vartype output_class_names: list of str
    :ivar class_groups: A list which elements are lists of input class names
        defining the corresponding output class. For example, [1][4] is the
        fifth input class corresponding to the second output class.
    :vartype class_groups: list of list of str
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ctransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a ClassReducer.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instatiate a ClassReducer
        """
        # Initialize from parent
        kwargs = ClassTransformer.extract_ctransf_args(spec)
        # Extract particular arguments of ClassReducer
        kwargs['output_class_names'] = spec.get('output_class_names', None)
        kwargs['class_groups'] = spec.get('class_groups', None)
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a ClassReducer.

        :param kwargs: The attributes for the ClassReducer
        """
        # Call parent init
        super().__init__(**kwargs)
        # Assign attributes
        self.output_class_names = kwargs.get('output_class_names', None)
        self.class_groups = kwargs.get('class_groups', None)
        # Automatically generate output class names from class groups
        if self.output_class_names is None and self.class_groups is not None:
            self.output_class_names = [
                f'C{i}' for i in range(len(self.class_groups))
            ]
        # Exception : Number of output classes distinct from class groups
        elif (
            self.output_class_names is not None
            and self.class_groups is not None
            and len(self.output_class_names) != len(self.class_groups)
        ):
            raise ClassTransformerException(
                f'ClassReducer received {len(self.output_class_names)} output '
                f'classes but {len(self.class_groups)} class groups.'
            )
        # Exception : No class groups given
        if self.class_groups is None:
            raise ClassTransformerException(
                'ClassReducer received no class groups.'
            )

    # ---  CLASS TRANSFORM METHODS  --- #
    # --------------------------------- #
    def transform(self, y, out_prefix=None):
        """
        The fundamental transformation logic defining the class reducer.

        See :class:`.ClassTransformer` and
        :meth:`class_transformer.ClassTransformer.transform`.
        """
        # Transform
        start = time.perf_counter()
        yout = np.zeros(y.shape)-1
        for i, class_group in enumerate(self.class_groups):  # For any outclass
            # Compute a mask of true iff classified as inclass in group
            mask = np.zeros(y.shape, dtype=bool)  # False by default
            for inclass in class_group:  # For each input class in the group
                mask = mask + (y == self.cti[inclass])  # mask OR inclass
            yout[mask] = i
        end = time.perf_counter()
        # Log transformation
        LOGGING.LOGGER.info(
            'ClassReducer transformed {m} points from {nin} classes to '
            '{nout} classes in {t:.3f} seconds.'.format(
                m=len(y),
                nin=self.num_classes,
                nout=len(self.class_groups),
                t=end-start
            )
        )
        # Report class reduction
        self.report(
            ClassReductionReport(
                original_class_names=self.input_class_names,
                yo=y,
                reduced_class_names=self.output_class_names,
                yr=yout,
                class_groups=self.class_groups
            ),
            out_prefix=out_prefix
        )
        # Plot class reduction
        if self.plot_path is not None:
            ClassReductionPlot(
                original_class_names=self.input_class_names,
                yo=y,
                reduced_class_names=self.output_class_names,
                yr=yout,
                path=self.plot_path
            ).plot(out_prefix=out_prefix)
        # Return
        return yout
