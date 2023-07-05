# ---   IMPORTS   --- #
# ------------------- #
from src.plot.plot import PlotException
from src.plot.mpl_plot import MplPlot
from src.plot.plot_utils import PlotUtils
from matplotlib import pyplot as plt
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class KFoldPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the evaluation of kfold procedures.
    See :class:`.KFoldEvaluation`

    :ivar X: See :meth:`kfold_evaluator.KFoldEvaluator.eval`
    :ivar sigma: See :class:`.KFoldEvaluation`
    :ivar metric_names: The name for each metric used to evaluate the
        k-fold procedure.
    :vartype metric_names: list
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, X, sigma, **kwargs):
        """
        Initialize/instantiate a KFoldPlot.

        :param kwargs: The attributes for the KFoldPlot.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of KFoldPlot
        self.X = X
        self.sigma = sigma
        self.metric_names = kwargs.get('metric_names', None)
        # Validate attributes
        if self.X is None:
            raise PlotException(
                'KFoldPlot with no evaluation matrix is not supported.'
            )

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot a grid where each cell contains a subplot representing the
        k-folding evaluation on a particular metric.

        See :meth:`plot.Plot.plot`.
        """
        # Determine rows and cols
        nplots = self.X.shape[1]
        nrows, ncols = PlotUtils.rows_and_cols_for_nplots(nplots)
        t = np.arange(1, self.X.shape[0]+1)  # 1,...,num_folds
        # Build figure
        fig = plt.figure(figsize=kwargs.get('figsize', (2+3*nrows, 1+3*ncols)))
        # Make each subplot
        for i in range(nplots):
            ax = fig.add_subplot(nrows, ncols, i+1)
            ax.plot(  # Plot the values for the metric i
                t, self.X[:, i], lw=3, color='black'
            )
            ax.plot(  # Plot the upper deviation line
                t, self.X[:, i]+self.sigma[i], lw=2, color='#AA0000'
            )
            ax.plot(  # Plot the lower deviation line
                t, self.X[:, i]-self.sigma[i], lw=2, color='#AA0000'
            )
            ax.fill_between(
                t, self.X[:, i]-self.sigma[i], self.X[:, i]+self.sigma[i],
                color='#FF0000', alpha=0.67
            )
            # Format axes
            ax.grid('both')
            ax.set_axisbelow(True)
            ax.tick_params(which='both', axis='both', labelsize=18)
            ax.set_xlabel('fold', fontsize=18)
            if self.metric_names is not None:
                ax.set_ylabel(self.metric_names[i], fontsize=18)
        # Format figure
        fig.suptitle(kwargs.get('suptitle', 'k-fold evaluation'), fontsize=20)
        fig.tight_layout()
        # Make the plot effective
        self.save_show_and_clear()
