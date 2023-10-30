# ---   IMPORTS   --- #
# ------------------- #
from src.plot.plot import PlotException
from src.plot.mpl_plot import MplPlot
from matplotlib import pyplot as plt
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldsDistributionPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the distribution of predicted or expected value  in the
    receptive fields.
    See :class:`.ReceptiveFieldsDistributionPlot`.

    :ivar y_rf: The expected value for each point for each receptive field.
    :vartype y_rf: :class:`np.ndarray`
    :ivar yhat_rf: The predicted value for each point for each receptive field.
    :vartype yhat_rf: :class:`np.ndarray`
    :ivar class_names: The names representing each class.
    :vartype class_names: list
    """
    def __init__(self, **kwargs):
        # Call parent's init
        super().__init__(**kwargs)
        # Assign member attributes
        self.y_rf = kwargs.get('y_rf', None)
        self.yhat_rf = kwargs.get('yhat_rf', None)
        if self.y_rf is None and self.yhat_rf is None:
            raise PlotException(
                'Receptive field distribution plot is not possible without '
                'at least the expected or predicted classes.'
            )
        self.class_names = kwargs.get('class_names', None)
        if self.class_names is None:
            raise PlotException(
                'Receptive field distribution plot needs to receive the '
                'class names at initialization.'
            )

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot the distribution of predicted and/or expected values.

        See :meth:`plot.Plot.plot`.
        """
        # Count
        yhat_count, yhat_rf_count = self.count(
            self.yhat_rf
        ) if self.yhat_rf is not None else (None, None)
        y_count, y_rf_count = self.count(
            self.y_rf
        ) if self.y_rf is not None else (None, None)
        # Determine how many subplots and in how many rows and columns
        nplots = 4 if yhat_count is not None and y_count is not None else 2
        nrows, ncols = 1, 2
        if nplots == 4:
            nrows, ncols = 2, 2
        # Build figure
        fig = plt.figure(figsize=((14, 5) if nplots == 2 else (14, 9)))
        # Determine bar position on x-axis
        x = np.arange(len((yhat_count if yhat_count is not None else y_count)))
        # Do the subplots
        axes = []
        if yhat_count is not None:
            self.plot_counts(
                x,
                fig,
                yhat_count,
                yhat_rf_count,
                axes=axes,
                count_title='Predictions',
                rf_count_title='RF with predictions',
                subplot_offset=len(axes),
                nrows=nrows,
                ncols=ncols
            )
        if y_count is not None:
            self.plot_counts(
                x,
                fig,
                y_count,
                y_rf_count,
                axes=axes,
                count_title='References',
                rf_count_title='RF with references',
                subplot_offset=len(axes),
                nrows=nrows,
                ncols=ncols
            )
        # Format axes
        for ax in axes:
            ax.tick_params(axis='both', which='both', labelsize=14)
            ax.tick_params(axis='x', which='both', labelrotation=90)
            ax.set_ylabel('Absolute frequency', fontsize=14)
            ax.grid('both')
            ax.set_axisbelow(True)
        # Format figure
        fig.suptitle(
            'Point-wise distributions in the receptive fields', fontsize=16
        )
        fig.tight_layout()
        # Make the plot effective
        self.save_show_and_clear(
            out_prefix=kwargs.get('out_prefix', None),
            logging=kwargs.get('logging', False)
        )

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def count(self, y):
        """
        Compute the absolute frequency and the count of receptive fields
        containing at least one class case.

        :param y: Either the predicted or the expected values for each
            receptive field. In other words, y[i] is the i receptive field
            such that y[i][j] is the value corresponding to the point j in the
            receptive field i.
        :type y: :class:`np.ndarray` or list
        :return: The absolute frequency considering all points and the count
            of how many receptive fields do contain at least one element of
            a given class. If a and b are the returned vectors, a[i] is the
            absolute frequency of class i and b[i] is the number of receptive
            fields that contain at least one point of class i.
        :rtype: tuple
        """
        # Determine class_nums from given class_names
        class_nums = np.array(
            [i for i in range(len(self.class_names))],
            dtype=int
        )
        # Count class distribution
        count, bins = np.histogram(
            y, bins=len(class_nums), range=(0, len(class_nums))
        )
        # Count how many receptive fields contain at least one class case
        rf_count = np.array([
            np.count_nonzero([np.any(rf_i == cidx) for rf_i in y])
            for cidx in class_nums
        ])
        # Return
        return count, rf_count

    def plot_counts(
        self,
        x,
        fig,
        count,
        rf_count,
        axes=None,
        count_title=None,
        rf_count_title=None,
        subplot_offset=0,
        nrows=1,
        ncols=2
    ):
        """
        Do the plots corresponding to a set of counts. There are two potential
        sets of counts, the one corresponding to predictions, and the one
        corresponding to expected values.

        :param x: The values for the x-axis of the plots.
        :type x: :class:`np.ndarray`
        :param fig: The matplotlib figure where the plots belong to.
        :type fig: :class:`mpl.figure.Figure`
        :param count: How many cases among all receptive fields. In other
            words, count[i] is the number of cases corresponding to class i.
        :type count: :class:`np.ndarray`
        :param rf_count: How many receptive fields contain at least one case
            for each class. In other words, rf_count[i] is the number of
            receptive fields that contain at least one point of class i.
        :type rf_count: :class:`np.ndarray`
        :param axes: List where the axes of the generated plots will be
            appended.
        :type axes: list
        :param count_title: The title for the subplot corresponding
            to count.
        :type: count_title: str
        :param rf_count_title: The title for the subplot corresponding to
            rf_count.
        :param subplot_offset: How many subplots skip before generating the
            new ones.
        :type subplot_offset: int
        :param nrows: How many rows of subplots.
        :type nrows: int
        :paran ncols: How many columns of subplots.
        :type ncols: int
        :return: Nothing at all, but the plots are made effective. If axes is
            not None, the corresponding axes are appended to the list.
        """
        # Plot distribution of classes
        ax = fig.add_subplot(nrows, ncols, 1+subplot_offset)
        ax.set_title(count_title, fontsize=14)
        ax.bar(
            x, count,
            tick_label=self.class_names,
            color='tab:blue', edgecolor='black', linewidth=1
        )
        axes.append(ax)
        # Plot distribution of receptive fields with at least one class case
        ax = fig.add_subplot(nrows, ncols, 2+subplot_offset)
        ax.set_title(rf_count_title, fontsize=14)
        ax.bar(
            x, rf_count,
            tick_label=self.class_names,
            color='tab:red', edgecolor='black', linewidth=1
        )
        axes.append(ax)
