# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ClassReductionReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to class reductions.
    See :class:`.Report`.
    See also :class:`.ClassReducer`.

    :ivar original_class_names: The names of the original classes.
    :vartype original_class_names: list of str
    :ivar yo: The original classification.
    :vartype yo: :class:`np.ndarray`
    :ivar reduced_class_names: The names of the reduced classes.
    :vartype reduced_class_names: list of str
    :ivar yr: The reduced classification.
    :vartype yr: :class:`np.ndarray`
    :ivar class_groups: List such that [i] is the list of original class
        names that were reduced to the reduced class i.
    :vartype class_groups: list of str
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of ClassReductionReport.

        :param kwargs: The key-word arguments defining the report's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the ClassReductionReport
        self.original_class_names = kwargs.get('original_class_names', None)
        self.yo = kwargs.get('yo', None)
        self.reduced_class_names = kwargs.get('reduced_class_names', None)
        self.yr = kwargs.get('yr', None)
        self.class_groups = kwargs.get('class_groups', None)
        # Validate
        if self.original_class_names is None:
            raise ReportException(
                'Cannot build class reduction report without the original '
                'class names.'
            )
        if self.yo is None:
            raise ReportException(
                "Cannot build class reduction report without the original "
                "classification."
            )
        if self.reduced_class_names is None:
            raise ReportException(
                'Cannot build class reduction report without the reduced '
                'class names.'
            )
        if self.yr is None:
            raise ReportException(
                'Cannot build class reduction report without the reduced '
                'classification.'
            )

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the report about a class reduction.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        # --- Introduction --- #
        s = 'Class reduction'
        if self.class_groups is not None:
            s += ' (reduced <- originals)'
        s += ':\n\n'
        if self.class_groups is not None:
            for i, class_group in enumerate(self.class_groups):
                s += f'{self.reduced_class_names[i]} <- {class_group}\n'
        s += '\n'
        # --- Content --- #
        s += self.to_class_distribution(  # Add original class distribution
            "ORIGINAL CLASSES",
            self.original_class_names,
            self.yo
        )
        s += '\n'
        s += self.to_class_distribution(  # Add reduced class distribution
            "REDUCED CLASSES",
            self.reduced_class_names,
            self.yr
        )
        # Return
        return s

    def to_class_distribution(self, title, class_names, y):
        """
        Generate a string representing a class distribution.

        :param title: The title or name for the class distribution
            representation.
        :type title: str
        :param class_names: The name for each class.
        :type class_names: list of str
        :param y: The class for each point.
        :type y: :class:`np.ndarray`
        :return: String representing a class distribution.
        """
        # --- Introduction --- #
        s = title
        # ---  Head  --- #
        s += "\nCLASS                   ,       ABS. COUNT,       PERCENTAGE\n"
        # ---  Body  --- #
        accum_percent = 0
        num_points = len(y)
        for class_idx, class_name in enumerate(class_names):
            class_count = np.count_nonzero(y == class_idx)
            class_percent = 100*class_count / num_points
            accum_percent += class_percent
            s += f'{class_name:24}, {class_count:16d}, {class_percent:16.3f}\n'
        s += f'TOTAL                   , '\
             f'{num_points:16d}, {accum_percent:16.3f}\n'
        # Return
        return s
