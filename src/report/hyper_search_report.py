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

    # ---   TO STRING   --- #
    # --------------------- #
    def __str__(self):
        """
        The string representation of the hyperparameter search report.
        See :class:`.Report` and also :meth:`report.Report.__str__`.
        """
        # Build header
        has_train_scores = \
            self.results.get('mean_train_score', None) is not None
        params = self.results['params']
        params_name = [key for key in params[0].keys()]
        s = ''
        for param_name in params_name:
            s += f'{param_name:16.16},'
        if has_train_scores:
            s += 'mean_train,std_train ,'
        s += 'mean_test ,std_test  ,mean_time ,std_time  '
        # Determine sort
        I = np.argsort(self.results["mean_test_score"])
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
                s += f'  {str(100*self.results["mean_train_score"][i]):8.8},'
                s += f'  {str(100*self.results["std_train_score"][i]):8.8},'
            s += f'  {str(100*self.results["mean_test_score"][i]):8.8},'
            s += f'  {str(100*self.results["std_test_score"][i]):8.8},'
            s += f'  {str(self.results["mean_fit_time"][i]):8.8},'
            s += f'  {str(self.results["std_fit_time"][i]):8.8}'
        # Return
        return s
