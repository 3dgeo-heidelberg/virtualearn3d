# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class BestScoreSelectionReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports based on selecting the best scoring features.
    See :class:`.Report`
    See also :class:`.KBestSelector` and :class:`.PercentileSelector`.


    :ivar fnames: The names of all the features.
    :vartype fnames: list
    :ivar scores: The feature-wise scores.
    :vartype scores: :class:`np.ndarray`
    :ivar score_name: The name of the score.
    :vartype score_name: str
    :ivar pvalues: The p-values, if available, i.e., can be None.
    :vartype pvalues: :class:`np.ndarray` or None
    :ivar selected_features: The indices or bool mask representing the selected
        features. This is an optional attribute, if it is not available, then
        selected features will not be reported.
    :vartype selected_features: list
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(
        self,
        fnames,
        scores,
        score_name,
        pvalues=None,
        selected_features=None,
        **kwargs
    ):
        """
        Initialize an instance of BestScoreSelectionReport.

        :param fnames: The names of all the features (OPTIONAL). If not given,
            the default will be f1, ..., fn.
        :param scores: The feature-wise scores.
        :param score_name: The name of the score.
        :param pvalues: The p-values (OPTIONAL).
        :param selected_features: The indices or bool mask representing the
            selected features (OPTIONAL).
        :param kwargs: The key-word arguments.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the BestScoreSelectionReport
        self.fnames = fnames
        self.scores = scores
        self.score_name = score_name
        self.pvalues = pvalues
        self.selected_features = selected_features
        # Validate
        if self.scores is None:
            raise ReportException(
                'BestScoreSelectionReport is not possible without scores.'
            )
        # Handle score_name when it is none
        if self.score_name is None:
            self.score_name = "SCORE"
        # Handle fnames when they are None
        if self.fnames is None:
            self.fnames = [f'f{i}' for i in range(1, len(self.scores)+1)]

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the best score selection report.
        See :class:`.Report` and also :meth:`report.Report.__str__`
        """
        # --- Introduction --- #
        s = f'Best score selection on {len(self.scores)} features'
        if self.selected_features is not None:
            if self.selected_features.dtype == bool:
                s += f', selecting {np.count_nonzero(self.selected_features)}'
            else:
                s += f', selecting {len(self.selected_features)}'
        s += ':\n\n'
        # --- Table: score, variance --- #
        s += f'FEATURE                  , {self.score_name.upper():12.12}'
        if self.pvalues is not None:
            s += f' , P-VALUE'
        s += '\n'
        var_sort = np.argsort(self.scores)
        for i in var_sort:
            fname = self.fnames[i]
            score = self.scores[i]
            pvalue = None if self.pvalues is None else self.pvalues[i]
            s += f'{fname:24.24} , {score:12.3f}'
            if pvalue is not None:
                s += f' , {pvalue:8.3f}'
            s += '\n'
        # --- Selected features --- #
        if self.selected_features is not None:
            s += '\n\nSELECTED FEATURES:\n'
            for fname in np.array(self.fnames)[self.selected_features]:
                s += f'{fname}\n'
        # Return
        return s
