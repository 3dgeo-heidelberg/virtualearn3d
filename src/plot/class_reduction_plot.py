# ---   IMPORTS   --- #
# ------------------- #
from src.plot.plot import PlotException
from src.plot.mpl_plot import MplPlot
import src.main.main_logger as LOGGING
from matplotlib import pyplot as plt
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ClassReductionPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the results of a class reduction task.

    See :class:`.MplPlot` and :class:`.ClassReducer`.

    :ivar original_class_names: The names of the original classes.
    :vartype original_class_names: list of str
    :ivar yo: The original classification.
    :vartype yo: :class:`np.ndarray`
    :ivar reduced_class_names: The names of the reduced classes.
    :vartype reduced_class_names: list of str
    :ivar yr: The reduced classification.
    :vartype yr: :class:`np.ndarray`
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of ClassReductionPlot.

        :param kwargs: The key-word arguments defining the plot's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of ClassReductionPlot
        self.original_class_names = kwargs.get('original_class_names', None)
        self.yo = kwargs.get('yo', None)
        self.reduced_class_names = kwargs.get('reduced_class_names', None)
        self.yr = kwargs.get('yr', None)
        # Validate
        if self.original_class_names is None:
            raise PlotException(
                'Cannot build class reduction plot without the original '
                'class names.'
            )
        if self.yo is None:
            raise PlotException(
                "Cannot build class reduction plot without the original "
                "classification."
            )
        if self.reduced_class_names is None:
            raise PlotException(
                'Cannot build class reduction plot without the reduced '
                'class names.'
            )
        if self.yr is None:
            raise PlotException(
                'Cannot build class reduction plot without the reduced '
                'classification.'
            )

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot the class distribution of the original and reduced classes.

        See :meth:`plot.Plot.plot`.
        """
        # Build figure
        fig = plt.figure(figsize=(14, 5))
        # Plot original class distribution
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('Original class distribution', fontsize=16)
        self.plot_class_distribution(ax, self.original_class_names, self.yo)
        # Plot reduced class distribution
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('Reduced class distribution', fontsize=16)
        self.plot_class_distribution(ax, self.reduced_class_names, self.yr)
        # Format the figure
        fig.tight_layout()
        # Make the plot effective
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))

    def plot_class_distribution(self, ax, class_names, y):
        """
        Plot the class distribution.

        :param ax: The axes where the plot must be drawn.
        :param class_names: The name for each class.
        :param y: The point-wise classes.
        :return: Nothing at all, but the subplot is drawn in the given axes.
        """
        # Determine bar position on x-axis
        x = np.arange(len(class_names))
        # Count points per classes
        y_count, y_bin = np.histogram(y, bins=len(class_names))
        # Plot the bars
        ax.bar(
            x, y_count, tick_label=class_names,
            edgecolor='black', linewidth=1
        )
        # Format the axes
        ax.tick_params(axis='both', which='both', labelsize=14)
        ax.tick_params(axis='x', which='both', labelrotation=90)
        ax.set_ylabel('Number of points', fontsize=14)
        ax.grid('both')
        ax.set_axisbelow(True)
