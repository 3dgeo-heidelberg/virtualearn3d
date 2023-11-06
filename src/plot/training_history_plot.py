# ---   IMPORTS   --- #
# ------------------- #
from src.plot.plot import PlotException
from src.plot.mpl_plot import MplPlot
from src.plot.plot_utils import PlotUtils
from matplotlib import pyplot as plt
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class TrainingHistoryPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot (potentially many plots) the training history of a deep
    learning model, i.e., neural networks.

    :ivar history: The history.
    :vartype history: :class:`tf.keras.callbacks.History`
    :ivar filter: The name of the filter to be applied (None means no filtering
        ).
    :vartype filter: str or None
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, history, **kwargs):
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of TrainingHistoryPlot
        self.history = history
        self.filter = kwargs.get('filter', None)
        # Validate attributes
        if self.history is None:
            raise PlotException(
                'TrainingHistoryPlot without training history is not '
                'supported. None was given.'
            )
        if self.history.history is None or len(self.history.history) < 1:
            raise PlotException(
                'TrainingHistoryPlot received an empty history. It is not '
                'supported.'
            )

    # ---  PLOT METHODS  --- #
    # ---------------------- #
    def plot(self, **kwargs):
        """
        Do the plots related to the training history.

        See :meth:`plot.Plot.plot`
        """
        # Store path
        _path = self.path
        # Determine epochs
        epochs = [
            i+1 for i in range(len(list(self.history.history.values())[0]))
        ]
        # Do individual plots
        for k, v in self.history.history.items():
            self.path = os.path.join(_path, f'{k}.svg')
            self.do_isolated_plot(epochs, k, v, **kwargs)
        self.path = _path
        # Do summary plot
        self.path = os.path.join(_path, f'summary.svg')
        self.do_summary_plot(epochs, **kwargs)
        # Restore path
        self.path = _path

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    def do_isolated_plot(self, epochs, name, values, **kwargs):
        """
        Method to do handle the plot for each metric in the history.

        :param epochs: The sequence of numbers representing the involved
            epochs, e.g., [0, 1, 2, 3, 4].
        :type epochs: list
        :param name: The name of the metric.
        :type name: str
        :param values: The values of the metric.
        :type values: list or tuple or :class:`np.ndarray`
        :param kwargs: The key-word arguments. See :meth:`plot.Plot.plot`.
        :return: Nothing at all, but the plot plot is exported.
        """
        # Build figure
        fig = plt.figure(figsize=(7, 5))
        # Make the plot
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(epochs, values, lw=3, color='tab:blue')
        # Format the plot
        self.format_plot(fig, ax, name, values)
        # Format the figure
        fig.tight_layout()
        # Make the plot effective
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix'))

    def do_summary_plot(self, epochs, **kwargs):
        """
        Method to do handle the summary plot representing all metrics in
        the history.

        :param epochs: The sequence of numbers representing the involved
            epochs, e.g., [0, 1, 2, 3, 4].
        :type epochs: list
        :param kwargs: The key-word arguments. See :meth:`plot.Plot.plot`.
        :return: Nothing at all, but the plot subplot is exported.
        """
        # Build figure
        fig = plt.figure(figsize=(15, 10))
        nplots = len(self.history.history)
        nrows, ncols = PlotUtils.rows_and_cols_for_nplots(nplots) \
            if nplots != 3 else (2, 2)
        # Make the plots
        colors = ['tab:red', 'tab:green', 'tab:blue']
        history_items = list(self.history.history.items())
        for i in range(nplots):
            key, values = history_items[i]
            ax = fig.add_subplot(nrows, ncols, i+1)
            ax.plot(epochs, values, lw=3, color=colors[i % len(colors)])
            self.format_plot(fig, ax, key, values)
        # Format the figure
        fig.tight_layout()
        # Make the plot effective
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix'))

    def format_plot(self, fig, ax, name, values):
        """
        Apply format to given plot.

        :param fig: The plot's figure.
        :param ax: The plot's axes.
        :param name: The y label for the plot.
        :param values: The plotted values.
        :return: Nothing at all, but the format of the input plot is updated.
        """
        # Filter values if requested
        vmin, vmax = self.filter_values(values)
        # The format itself
        ax.set_ylim(
            vmin-0.01*(vmax-vmin),
            vmax+0.01*(vmax-vmin)
        )
        ax.grid('both')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='both', labelsize=18)
        ax.set_xlabel("epochs", fontsize=18)
        ax.set_ylabel(name, fontsize=18)

    def filter_values(self, values):
        r"""
        Filter given values. There are three possible filter modes.

        **Filter: None**

        No filter is returned.

        **Filter: "quartile"**

        The
        :math:`[Q_1-\frac{3}{2}\mathrm{IQR}, Q_3+\frac{3}{2}\mathrm{IQR}]`
        filter is returned.

        **Filter: "stdev"**

        The :math:`[\mu-3\sigma, mu+3\sigma]` filter is
        returned. Where :math:`\mu` is the mean and :math:`\sigma` is the
        standard deviation.

        :param values: The values to be filtered.
        :type values: tuple or list or :class:`np.ndarray`
        :return: The min and max values defining the filtering interval.
        :rtype: tuple
        """
        if self.filter is None:
            return np.min(values), np.max(values)
        elif self.filter.lower() == 'quartile':
            Q = np.quantile(values, [1/4, 2/4, 3/4])
            IQR = Q[2] - Q[0]
            return Q[0]-3/2*IQR, Q[2]+3/2*IQR
        elif self.filter.lower() == 'stdev':
            mu, sigma = np.mean(values), np.std(values)
            return mu-3*sigma, mu+3*sigma
        else:
            raise PlotException(
                'TrainingHistoryPlot is configured with an unexpected filter '
                f'specification: "{self.filter}".'
            )
