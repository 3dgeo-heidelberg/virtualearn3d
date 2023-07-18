# ---   IMPORTS   --- #
# ------------------- #
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class PlotUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods for common plot operations.
    """
    # ---   GRID OF FIGURES   --- #
    # --------------------------- #
    @staticmethod
    def rows_and_cols_for_nplots(nplots, transpose=False):
        """
        Determine the number of rows and columns required to plot a grid of
        n subplots.

        By default, the number of rows is greater than or equal to the number
        of columns. This condition can be reversed by setting transpose to
        True.

        :param nplots: The number of subplots in the grid (n).
        :param transpose: If True, return cols and rows instead of rows and
            cols.
        :return: The number of rows and the number of columns required to plot
            n subplots.
        :rtype: (int, int)
        """
        # Determine number of rows and columns
        ncols = int(np.sqrt(nplots))
        nrows = int(np.ceil(nplots/ncols))
        # Return
        if transpose:
            return ncols, nrows
        return nrows, ncols
