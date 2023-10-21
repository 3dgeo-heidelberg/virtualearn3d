# ---   IMPORTS   --- #
# ------------------- #
from src.plot.mpl_plot import MplPlot
import src.main.main_logger as LOGGING
from matplotlib import pyplot as plt
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class FeaturesStructuringLayerPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the insights of a features structuring layer.

    See :class:`.MplPlot` and :class:`.FeaturesStructuringLayer`.

    :ivar omegaD: The vector of distance weights.
    :vartype omegaD: :class:`np.ndarray`
    :ivar omegaF: The vector of feature weights.
    :vartype omegaF: :class:`np.ndarray`
    :ivar xmax: The maximum value for the distance domain (defines the x-axis
        for omegaD plots).
    :vartype xmax: float
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of FeaturesStructuringLayerPlot.

        :param kwargs: The key-word arguments defining the plot's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of FeaturesStructuringLayerPlot
        self.omegaD = kwargs.get('omegaD', None)
        self.omegaF = kwargs.get('omegaF', None)
        self.xmax = kwargs.get('xmax', 1)

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot the omegaD(x) for x in [0, xmax] as a function and omegaF as bars.

        See :meth:`plot.Plot.plot`.
        """
        # Build figure
        fig = plt.figure(figsize=(14, 7))
        # Handle omegaD(x) plot
        if self.omegaD is not None:
            ax = fig.add_subplot(1, 2, 1)
            x = np.linspace(0, self.xmax, 300)
            for omegaDi in self.omegaD:
                dQ = np.exp(-(x*x)/(omegaDi*omegaDi))
                ax.plot(x, dQ, lw=1)
            ax.set_title(
                'Kernel distance for each $\\omega_{Di}$',
                fontsize=14
            )
            ax.set_xlabel('$x$ (distance)', fontsize=12)
            ax.set_ylabel('$d_Q(x)$', fontsize=12)
            ax.grid('both')
            ax.set_axisbelow(True)
        # Handle omegaF bars plot
        if self.omegaF is not None:
            ax = fig.add_subplot(1, 2, 2)
            x = np.arange(0, len(self.omegaF), dtype=int)
            ax.bar(x, self.omegaF, edgecolor='black', linewidth=0.5)
            ax.set_title('$\\omega_F$', fontsize=14)
            ax.set_ylabel('$\\omega_{Fi}$', fontsize=12)
            ax.set_xlabel('$i$', fontsize=12)
            ax.grid('both')
            ax.set_axisbelow(True)
        # Format figure
        fig.tight_layout()
        # Make plot effective
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
