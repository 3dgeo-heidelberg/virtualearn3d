# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class HyperSearchReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports related to search-based hyperparameter tuning.
    See :class:`.Report`.
    See also :class:`.HyperGridSearch` and :class:`.HyperRandomSearch`.

    :ivar results: The results from a hyperparameter tuning optimization.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, results, **kwargs):
        """
        Initialize an instance of HyperSearchReport.

        :param results: The results from a hyperparameter tuning optimization.
        :param kwargs: The key-word arguments defining the report's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Basic attributes of the HyperSearchReport
        self.results = results
        self.scores = kwargs.get('scores')
        # Transform scores to list
        if isinstance(self.scores, dict):
            self.scores = list(self.scores.keys())
        if isinstance(self.scores, str):
            self.scores = [self.scores]

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the hyperparameter search report.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        # Extract scores
        mean_test_scores_keys = [
            x for x in self.results if x.lower().find('mean_test') >= 0
        ]
        mean_test_scores = [self.results[key] for key in mean_test_scores_keys]
        std_test_scores_keys = [
            x for x in self.results if x.lower().find('std_test') >= 0
        ]
        std_test_scores = [self.results[key] for key in std_test_scores_keys]
        mean_train_scores_keys = [
            x for x in self.results if x.lower().find('mean_train') >= 0
        ]
        mean_train_scores = [self.results[key] for key in mean_train_scores_keys]
        std_train_scores_keys = [
            x for x in self.results if x.lower().find('std_test') >= 0
        ]
        std_train_scores = [self.results[key] for key in std_train_scores_keys]
        # TODO Pending : Handling of training scores is implemented here but
        # not supported. There is no urgent need to support them for now.
        # Build header
        has_train_scores = len(mean_train_scores_keys) > 0
        params = self.results['params']
        params_name = [key for key in params[0].keys()]
        s = ''
        for param_name in params_name:
            s += f'{param_name:16.16},'
        if has_train_scores:
            for i in range(len(mean_train_scores_keys)):
                s += f'{mean_train_scores_keys[i]:>16.16},'\
                     f'{std_test_scores_keys[i]:>16.16},'
        for i in range(len(mean_test_scores_keys)):
            s += f'{mean_test_scores_keys[i]:>16.16},'\
                 f'{std_test_scores_keys[i]:>16.16},'
        s += ' mean_time,  std_time'
        # Determine sort
        I = np.argsort(self.results[mean_test_scores_keys[0]])
        # Populate body
        nrows = len(params)
        for jrow in range(nrows):
            i = I[jrow]
            paramsi = params[i]
            s += '\n'
            for param_key in paramsi.keys():
                if isinstance(paramsi[param_key], str):
                    s += f'{paramsi[param_key]:16.16},'
                else:
                    s += f'{str(paramsi[param_key]):8.8}        ,'
            if has_train_scores:
                for k in range(len(mean_train_scores_keys)):
                    s += f'{100*mean_train_scores[k][i]:16.3f},'
                    s += f'{100*std_train_scores[k][i]:16.3f},'
            for k in range(len(mean_test_scores_keys)):
                s += f'{100*mean_test_scores[k][i]:16.3f},'
                s += f'{100*std_test_scores[k][i]:16.3f},'
            s += f'  {self.results["mean_fit_time"][i]:8.3f},'
            s += f'  {self.results["std_fit_time"][i]:8.3f}'
        # Return
        return s
