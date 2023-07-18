# ---   IMPORTS   --- #
# ------------------- #
from abc import ABC
from src.plot.plot import Plot
from src.inout.io_utils import IOUtils
from matplotlib import pyplot as plt
import os


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
        to be called at the end of a plot method invocation. See
        :meth:`plot.Plot.plot`.

        :return: Nothing.
        """
        # Save
        if self.path:
            path = self.path
            if out_prefix is not None:
                path = out_prefix[:-1] + path[1:]
            IOUtils.validate_path_to_directory(
                os.path.dirname(path),
                'It is not possible to write a plot to the given path: '
                f'"{path}"'
            )
            plt.savefig(path)
        # Show
        if self.show:
            plt.show()
        # Clear
        plt.gcf()
        plt.close()
