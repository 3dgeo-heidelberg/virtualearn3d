# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException


# ---   EXCEPTIONS   --- #
# ---------------------- #
class PlotException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to plot components.
    See :class:`.VL3DException`
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class Plot:
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing the interfaces governing any plot.

    :ivar path: The path to save the plot to a file.
    :vartype path: str
    :ivar show: The boolean flag controlling whether to show the plot (True)
        or not (False).
    :vartype show: bool
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Root initialization for any instance of type Plot.

        :param kwargs: The attributes for the plot.
        """
        # Fundamental plot initialization
        self.path = kwargs.get('path', None)
        self.show = kwargs.get('show', False)

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    @abstractmethod
    def plot(self, **kwargs):
        """
        Do the plot effective. The typical steps are:

        1. Do plot computations, i.e., prepare the plot, build the plot, and
        format the plot).

        2. If the plot is associated to a path, then write it to the
        corresponding file.

        3. If the show flag is true, visualize the plot.

        :return: Nothing.
        """
        pass
