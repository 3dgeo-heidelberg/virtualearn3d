# ---   IMPORTS   --- #
# ------------------- #
from src.plot.mpl_plot import MplPlot
import src.main.main_logger as LOGGING
from matplotlib import pyplot as plt
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class ClassificationUncertaintyPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the uncertainty of classified point clouds.

    See :class:`.MplPlot` and :class:`.ClassificationUncertaintyEvaluation`.

    :ivar class_names: See :class:`.ClassificationUncertaintyEvaluation`.
    :ivar y: See :class:`.ClassificationUncertaintyEvaluation`.
    :ivar yhat: See :class:`.ClassificationUncertaintyEvaluation`.
    :ivar pwise_entropy: See :class:`.ClassificationUncertaintyEvaluation`.
    :ivar weighted_entropy: See :class:`.ClassificationUncertaintyEvaluation`.
    :ivar cluster_wise_entropy: See :class:`.ClassificationUncertaintyEvaluation`.
    :ivar class_ambiguity: See :class:`.ClassificationUncertaintyEvaluation`.
    :ivar gaussian_kernel_points: See :class:`.ClassificationUncertaintyEvaluation`.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of ClassificationUncertaintyPlot.

        :param kwargs: The key-word arguments defining the plot's attributes.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of ClassificationUncertaintyPlot
        self.class_names = kwargs.get('class_names', None)
        self.y = kwargs.get('y', None)
        self.yhat = kwargs.get('yhat', None)
        self.pwise_entropy = kwargs.get('pwise_entropy', None)
        self.weighted_entropy = kwargs.get('weighted_entropy', None)
        self.cluster_wise_entropy = kwargs.get('cluster_wise_entropy', None)
        self.class_ambiguity = kwargs.get('class_ambiguity', None)
        self.gaussian_kernel_points = kwargs.get('gaussian_kernel_points', 256)

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot representations of the available uncertainty metrics.

        See :meth:`plot.Plot.plot`.
        """
        # Path expansion, if necessary
        _path = self.path
        prefix = kwargs.get('out_prefix', None)
        if prefix is not None:
            prefix = prefix[:-1]  # Ignore '*' at the end
            _path = prefix + _path[1:]
        # Measure time : Start
        start = time.perf_counter()
        # Prepare list of uncertainty metrics, names, and path suffixes
        _E, _Enames, _Esuffixes = [], [], []
        if self.pwise_entropy is not None:
            _E.append(self.pwise_entropy)
            _Enames.append('Point-wise entropy')
            _Esuffixes.append('point_wise_entropy_figure')
        if self.weighted_entropy is not None:
            _E.append(self.weighted_entropy)
            _Enames.append('Weighted entropy')
            _Esuffixes.append('weighted_entropy_figure')
        if self.cluster_wise_entropy is not None:
            _E.append(self.cluster_wise_entropy)
            _Enames.append('Cluster-wise entropy')
            _Esuffixes.append('cluster_wise_entropy_figure')
        if self.class_ambiguity is not None:
            _E.append(self.class_ambiguity)
            _Enames.append('Class ambiguity')
            _Esuffixes.append('class_ambiguity_figure')
        for k in range(len(_E)):
            # Plot point-wise entropies
            E = _E[k]
            Ename = _Enames[k]
            # Prepare figure
            fig = plt.figure(figsize=(16, 10))
            # Full histogram subplot
            ax = fig.add_subplot(2, 2, 1)
            self.plot_full_histogram(fig=fig, ax=ax, Ename=Ename, E=E)
            # Hit-fail histogram subplot
            ax = fig.add_subplot(2, 2, 2)
            self.plot_hitfail_histogram(fig=fig, ax=ax, Ename=Ename, E=E)
            # Reference-wise subplot
            ax = fig.add_subplot(2, 2, 3)
            self.plot_classwise_violin(
                fig=fig, ax=ax, Ename=Ename, E=E, y=self.y,
                title=f'{Ename} by reference class'
            )
            # Prediction-wise subplot
            ax = fig.add_subplot(2, 2, 4)
            self.plot_classwise_violin(
                fig=fig, ax=ax, Ename=Ename, E=E, y=self.yhat,
                title=f'{Ename} by predicted class'
            )
            # Format figure
            fig.tight_layout()
            # Make plot effective
            path = self.path
            self.path = path + _Esuffixes[k] + '.svg'
            self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
            self.path = path
        # Measure time : End
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'ClassificationUncertaintyPlot generated and wrote figures to '
            f'"{_path}" in {end-start:.3f} seconds.'
        )

    def plot_full_histogram(self, fig=None, ax=None, Ename=None, E=None):
        """
        Handle the full histogram subplot.

        :param fig: The figure context.
        :param ax: The axes context.
        :param Ename: The name of the entropy or uncertainty metric.
        :param E: The entropy or uncertainty metric.
        """
        # Full histogram subplot
        ax.set_title(f'{Ename} distribution')
        Eweights = 100*np.ones_like(E)/len(E)
        ax.hist(E, bins=32, weights=Eweights)
        ax.set_xlabel(Ename, fontsize=12)
        ax.set_ylabel('Percentage ($\\%$)', fontsize=12)
        ax.tick_params(axis='both', which='both', labelsize=12)
        ax.grid('both')
        ax.set_axisbelow(True)

    def plot_hitfail_histogram(self, fig=None, ax=None, Ename=None, E=None):
        """
        Handle the hit/fail histogram subplot.

        :param fig: The figure context.
        :param ax: The axes context.
        :param Ename: The name of the entropy or uncertainty metric.
        :param E: The entropy or uncertainty metric.
        """
        # Hit-fail histogram subplot
        ax.set_title(f'{Ename} success/fail distribution')
        if self.y is not None and self.yhat is not None:
            Ehit = E[self.y == self.yhat]
            Efail = E[self.y != self.yhat]
            hit_weights = 100*np.ones_like(Ehit)/len(Ehit)
            fail_weights = 100*np.ones_like(Efail)/len(Efail)
            ax.hist(
                [Ehit, Efail],
                weights=[hit_weights, fail_weights],
                bins=16,
                histtype='bar',
                label=['Success', 'Fail'],
                color=['tab:green', 'tab:red']
            )
            ax.legend(loc='best', fontsize=12)
            ax.set_xlabel(Ename, fontsize=12)
            ax.set_ylabel('Percentage ($\\%$)', fontsize=12)
            ax.tick_params(axis='both', which='both', labelsize=12)
            ax.grid('both')
            ax.set_axisbelow(True)

    def plot_classwise_violin(
        self, fig=None, ax=None, Ename=None, E=None, y=None, title=None
    ):
        """
        Handle the class-wise violin subplot.

        :param fig: The figure context.
        :param ax: The axes context.
        :param Ename: The name of the entropy or uncertainty metric.
        :param E: The entropy or uncertainty metric.
        :param y: The labels (either reference labels or predicted labels).
        """
        if title is not None:
            ax.set_title(title)
        if y is not None:
            violin_dataset = [
                E[y == class_idx]
                for class_idx in range(len(self.class_names))
            ]
            class_names = [class_name for class_name in self.class_names]
            for i in range(len(self.class_names)-1, -1, -1):
                if len(violin_dataset[i]) == 0:
                    LOGGING.LOGGER.debug(
                        'ClassificationUncertaintyPlot excludes '
                        f'{class_names[i]} from {Ename} violin plot because '
                        'it has no matches.'
                    )
                    del violin_dataset[i]
                    del class_names[i]
            parts = ax.violinplot(
                violin_dataset,
                showmedians=True,
                points=min(self.gaussian_kernel_points, y.shape[0]),
                widths=0.75
            )
            for polycol in parts['bodies']:
                polycol.set_facecolor('goldenrod')
                polycol.set_edgecolor('black')
                polycol.set_alpha(0.67)
                polycol.set_zorder(5)
            parts['cmins'].set_color('black')
            parts['cmins'].set_linewidth(3)
            parts['cmins'].set_zorder(6)
            parts['cmaxes'].set_color('black')
            parts['cmaxes'].set_linewidth(3)
            parts['cmaxes'].set_zorder(6)
            parts['cbars'].set_color('black')
            parts['cbars'].set_linewidth(1)
            parts['cbars'].set_zorder(6)
            parts['cmedians'].set_color('tab:blue')
            parts['cmedians'].set_linewidth(2)
            parts['cmedians'].set_zorder(7)
            parts['cmedians'].set_label('median')
            ax.set_xticks(
                [i+1 for i in range(len(violin_dataset))],
                labels=class_names
            )
            ax.set_ylabel(Ename, fontsize=12)
            ax.tick_params(axis='both', which='both', labelsize=12)
            ax.tick_params(axis='x', which='both', labelrotation=90)
            ax.grid('both')
            ax.set_axisbelow(True)
            ax.legend(loc='best', fontsize=12).set_zorder(10)
