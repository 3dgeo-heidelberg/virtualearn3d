# ---   IMPORTS   --- #
# ------------------- #
from src.plot.mpl_plot import MplPlot
from src.plot.plot import PlotException
from matplotlib import pyplot as plt
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class FeatureProcessingLayerPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the insights of a feature processing layer.

    See :class:`.MplPlot` :class:`.FeatureProcessingLayer`.

    :ivar M: The matrix of kernel's centers.
    :vartype M: :class:`np.ndarray`
    :ivar Omega: The matrix of kernel sizes (think about curvatures).
    :vartype Omega: :class:`np.ndarray`
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, M, Omega, **kwargs):
        """
        Initialize an instance of FeatureProcessingLayerPlot

        :param kwargs: The key-word arguments defining the plot's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.M = M
        self.Omega = Omega
        self.kernel_type = kwargs.get('kernel_type', 'Gaussian')
        # Validate
        if self.M is None or len(self.M) < 1:
            raise PlotException(
                'FeatureProcessingLayerReport did not receive the centers of '
                'the kernel.'
            )
        if self.Omega is None or len(self.Omega) < 1:
            raise PlotException(
                'FeatureProcessingLayerReport did not receive the sizes of '
                'the kernel.'
            )

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot the kernel's radial basis functions.

        For each feature, the domain of the variable :math:`x` is considered
        as :math:`x \in [x_*, x^*]` where
        :math:`x_* = \min_{1 \leq i \leq K} \; \{x_i\} - \sigma` and
        :math:`x^* = \max_{1 \leq 1 \leq K} \; \{x_i\} + \sigma` with
        :math:`\sigma` the standard deviation of the variable.

        See :meth:`plot.Plot.plot`.
        """
        def plot_rbf(ax, x, y):
            ax.plot(x, y, zorder=4)
        # Build figure
        n_rbfs = self.M.shape[0]
        n_feats = self.M.shape[1]
        base_nrows = 3  # How many rows in the figure, assuming unitary res.
        base_ncols = 6  # How many columns in the figure, assuming unitary res.
        fig_res = max(1, n_feats//(base_nrows*base_ncols))  # Fig. resolution
        ncols = base_ncols  # Final number of columns
        nrows = int(np.ceil(n_feats/ncols))  # Final number of rows
        fig_width, fig_height = 20*fig_res, 15*fig_res  # Figure size
        fig = plt.figure(figsize=(fig_width, fig_height))  # Make figure
        # Do one subplot for each RBF
        kernel_type = self.kernel_type.lower()
        for j in range(n_feats):
            # Prepare axes
            xstdev = np.std(self.M[:, j])
            xmin, xmax = np.min(self.M[:, j])-xstdev, np.max(self.M[:, j])+xstdev
            x = np.linspace(xmin, xmax, 300)
            ax = fig.add_subplot(nrows, ncols, j+1)
            # Plot RBFs
            y_sum = np.zeros_like(x)
            for i in range(n_rbfs):
                if kernel_type == 'gaussian':
                    y = np.exp(
                        -np.square(x-self.M[i, j])/np.square(self.Omega[i, j])
                    )
                elif kernel_type == 'markov':
                    y = np.exp(
                        -np.abs(x-self.M[i, j])/np.square(self.Omega[i, j])
                    )
                else:
                    raise PlotException(
                        'FeatureProcessingLayerPlot received an unexpected '
                        f'kernel type: "{self.kernel_type}"'
                    )
                y_sum = y_sum + y
                plot_rbf(ax, x, y)
            # Plot sum of RBFs
            y_sum = y_sum / n_rbfs
            ax2 = ax.twinx()
            ax2.plot(x, y_sum, lw=2, color='black', zorder=5)
        # Format figure
        fig.tight_layout()
        # Make plot effective
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
