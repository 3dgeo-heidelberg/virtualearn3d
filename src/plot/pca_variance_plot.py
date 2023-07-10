# ---   IMPORTS   --- #
# ------------------- #
from src.plot.plot import PlotException
from src.plot.mpl_plot import MplPlot
from src.plot.plot_utils import PlotUtils
from matplotlib import pyplot as plt
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class PCAVariancePlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the explained variance ratio from a PCA.
    See :class:`.PCATransformer`

    :ivar evr: The explained variance ratio for each feature derived by PCA
        projection.
    :vartype evr: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, evr, **kwargs):
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of PCAVariancePlot
        self.evr = evr
        # Validate attributes
        if self.evr is None:
            raise PlotException(
                'PCAVariancePlot with no explained variance ratios is not '
                'supported.'
            )

    # ---  PLOT METHODS  --- #
    # ---------------------- #
    def plot(self, **kwargs):
        """
        Plot the explained variance ratio (y axis) over the number of
        output dimensions (x axis).

        See :meth:`plot.Plot.plot`.
        """
        # Determine sequence of output dimensions
        dims = np.arange(1, len(self.evr)+1)  # 1, ... ,dim_out
        # Build figure
        fig = plt.figure(figsize=kwargs.get('figsize', (7, 5)))
        # Make the plot
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(  # Plot evr over out dims
            dims,
            np.cumsum(self.evr),
            color='tab:blue',
            lw=2
        )
        # Format axes
        ax.grid('both')
        ax.set_axisbelow(True)
        ax.set_xlabel('Output dimensionality (num. features)', fontsize=18)
        ax.set_ylabel('Explained Variance Ratio', fontsize=18)
        ax.tick_params(which='both', axis='both', labelsize=18)
        # Format figure
        fig.tight_layout()
        # Make the plot effective
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
