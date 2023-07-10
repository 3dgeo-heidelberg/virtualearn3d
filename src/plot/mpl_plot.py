# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.plot.plot import Plot
from matplotlib import pyplot as plt


# ---   CLASS   --- #
# ----------------- #
class MplPlot(Plot, ABC):
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing common behaviors for plots based on Matplotlib.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize a MplPlot.

        :param kwargs: The attributes for the MplPlot.
        """
        # Call parent's init
        super().__init__(**kwargs)

    # ---   MPL METHODS   --- #
    # ----------------------- #
    def save_show_and_clear(self, out_prefix=None):
        """
        Method to handle the save, show, and clear figure logic. It is expected
        to be called at the end of a plot method invokation. See
        :meth:`plot.Plot.plot`.

        :return: Nothing.
        """
        # Save
        if self.path:
            path = self.path
            if out_prefix:
                path = out_prefix[:-1] + path[1:]
            plt.savefig(path)
        # Show
        if self.show:
            plt.show()
        # Clear
        plt.gcf()
        plt.close()
