# ---   IMPORTS   --- #
# ------------------- #
from src.plot.mpl_plot import MplPlot
import src.main.main_logger as LOGGING
from matplotlib import pyplot as plt
from matplotlib import patheffects
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ClassificationPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the evaluation of a classification task.

    See :class:`.MplPlot` and :class:`.ClassificationEvaluation`.

    :ivar class_names: See :class:`.ClassificationEvaluation`.
    :ivar yhat_count: See :class:`.ClassificationEvaluation`.
    :ivar y_count: See :class:`.ClassificationEvaluation`.
    :ivar conf_mat: See :class:`.ClassificationEvaluation`.
    :ivar class_distribution_path: The path where the class distribution plot
        must be written. Can be None. In that case, no class distribution
        plot will be written.
    :vartype class_distribution_path: str
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of ClassificationPlot.

        :param kwargs: The key-word arguments defining the plot's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of ClassificationPlot
        self.class_names = kwargs.get('class_names', None)
        self.yhat_count = kwargs.get('yhat_count', None)
        self.y_count = kwargs.get('y_count', None)
        self.conf_mat = kwargs.get('conf_mat', None)
        self.class_distribution_path = kwargs.get('class_distribution_path')

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot the confusion matrix and the class distribution if the information
        is available.

        See :meth:`plot.Plot.plot`.
        """
        # Prepare path expansion, if necessary
        prefix = kwargs.get('out_prefix', None)
        if prefix is not None:
            prefix = prefix[:-1]  # Ignore '*' at the end
        # Handle confusion matrix plot
        if self.has_confusion_matrix():
            self.plot_confusion_matrix(**kwargs)
            path = self.path
            if prefix is not None:
                path = prefix + path[1:]
            LOGGING.LOGGER.info(
                f'ClassificationPlot wrote confusion matrix to "{path}"'
            )
        else:
            LOGGING.LOGGER.debug(
                'ClassificationPlot did NOT plot the confusion matrix.'
            )
        # Handle class distribution plot
        if self.has_class_distribution():
            self.plot_class_distribution(**kwargs)
            path = self.class_distribution_path
            if prefix is not None:
                path = prefix + path[1:]
            LOGGING.LOGGER.info(
                f'ClassificationPlot wrote class distribution to "{path}"'
            )
        else:
            LOGGING.LOGGER.debug(
                'ClassificationPlot did NOT plot the class distribution.'
            )

    def plot_confusion_matrix(self, **kwargs):
        """
        Plot the confusion matrix.
        """
        # Build figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1)
        # Plot confusion matrix
        conf_mat = self.conf_mat
        if len(self.class_names) > len(conf_mat):
            conf_mat = np.pad(
                conf_mat,
                pad_width=[0, len(self.class_names)-conf_mat.shape[0]]
            )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat,
            display_labels=self.class_names,
        )
        disp.plot(
            ax=ax,
            cmap='Greens',
            text_kw={
                'fontsize': 14,
                'fontweight': 'bold',
                'color': 'black',
                'path_effects': [
                    patheffects.withStroke(linewidth=3, foreground='white')
                ]
            },
            colorbar=False
        )
        # Color bar
        cbar = fig.colorbar(disp.im_)
        cbar.ax.tick_params(labelsize=14)
        # Format axes
        ax.tick_params(
            axis='both', which='both', labelsize=14, length=5.0, width=2.0
        )
        ax.tick_params(
            axis='x', which='both', labelrotation=90
        )
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)  # Plot border size
        # Format figure
        fig.tight_layout()
        # Make plot effective
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))

    def plot_class_distribution(self, **kwargs):
        """
        Plot the class distribution.
        """
        # Build figure
        fig = plt.figure(figsize=(14, 5))
        # Determine bar position on x-axis
        x = np.arange(len(self.yhat_count))
        # Plot predictions distribution
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('Predictions', fontsize=14)
        ax.bar(
            x, self.yhat_count,
            tick_label=self.class_names, edgecolor='black', linewidth=1
        )
        axes = [ax]
        # Plot references distribution
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('Reference', fontsize=14)
        ax.bar(
            x, self.y_count,
            tick_label=self.class_names, edgecolor='black', linewidth=1
        )
        axes.append(ax)
        # Format axes
        for ax in axes:
            ax.tick_params(axis='both', which='both', labelsize=14)
            ax.tick_params(axis='x', which='both', labelrotation=90)
            ax.set_ylabel('Number of points', fontsize=14)
            ax.grid('both')
            ax.set_axisbelow(True)
        # Format figure
        fig.suptitle('Point-wise class distributions', fontsize=16)
        fig.tight_layout()
        # Make plot effective
        path = self.path  # Store confusion matrix plot path
        self.path = self.class_distribution_path  # Replace path temporary
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
        self.path = path  # Restore confusion matrix plot path

    # ---  CHECK METHODS  --- #
    # ----------------------- #
    def has_confusion_matrix(self):
        """
        Check whether the plot contains a confusion matrix.

        :return: True if the plot contains a confusion matrix, False otherwise.
        """
        return self.conf_mat is not None and self.class_names is not None

    def has_class_distribution(self):
        """
        Check whether the plot contains all the information needed to plot the
        class distribution.

        :return: True if the plot contains all the information needed to plot
            the class distribution.
        """
        return (
            self.class_names is not None and
            self.yhat_count is not None and
            self.y_count is not None
        )
