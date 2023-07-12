# ---   IMPORTS   --- #
# ------------------- #
from src.plot.plot import PlotException
from src.plot.mpl_plot import MplPlot
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class RandForestPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the evaluation of trained random forest models.
    See :class:`.RandomForestEvaluation`.

    :ivar trees: The trees to be plotted.
    :vartype trees: list
    :ivar max_depth: The maximum depth to consider when plotting a tree. It is
        optional. When not given, full trees will be plotted.
    :vartype max_depth: int or None
    :ivar fnames: The feature names.
    :vartype fnames: list
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, trees, **kwargs):
        """
        Initialize/instantiate a RandForestPlot.

        :param trees: The tree-like estimators.
        :param fnames: The feature names.
        :param kwargs: The attributes for the RandForestPlot.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of RandForestPlot
        self.trees = trees
        self.max_depth = kwargs.get('max_depth', 5)
        self.fnames = kwargs.get('fnames', None)
        # Validate attributes
        if self.trees is None or len(self.trees) < 1:
            raise PlotException(
                'RandForestPlot with no trees is not supported.'
            )

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Do one plot per tree-like estimator representing the many decisions
        defining the model.

        See :meth:`plot.Plot.plot`.
        """
        # Single-tree case
        if len(self.trees) == 1:
            self.plot_tree(self.trees[0], 0)
        else:
            # Store path and index of last dot (before extension)
            old_path = self.path
            last_dot = old_path.rindex('.')
            # One plot per tree
            for i, tree in enumerate(self.trees):
                self.path = old_path[:last_dot]
                self.path += f'_{i+1}'
                self.path += old_path[last_dot:]
                self.plot_tree(tree, i)
            # Restore path
            self.path = old_path

    def plot_tree(self, tree, i):
        """
        Do the plot for a single tree.

        :param tree: The tree to be plotted.
        :param i: The number of the tree. NOTE it is not the index of the tree
            in the ensemble, unless all trees are considered. Instead, it is
            the number of the tree in the local context of selected trees.
        """
        fig = plt.figure()
        plot_tree(
            tree,
            max_depth=self.max_depth,
            filled=True,
            feature_names=self.fnames
        )
        fig.suptitle(f'Decision tree {i+1}', fontsize=14)
        fig.tight_layout()
        self.save_show_and_clear()
