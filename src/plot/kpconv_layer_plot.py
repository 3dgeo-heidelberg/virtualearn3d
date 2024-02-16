# ---   IMPORTS   --- #
# ------------------- #
from src.plot.mpl_plot import MplPlot
from src.plot.plot import PlotException
from src.plot.plot_utils import PlotUtils
import src.main.main_logger as LOGGING
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class KPConvLayerPlot(MplPlot):
    """
    :author: Alberto M. Esmoris Pena

    Class to plot the insights of a KPConv layer.

    See :class:`.KPConvLayerPlot` and :class:`.KPConvLayer`.

    :ivar Q: The matrix representing the kernel's structure.
    :vartype Q: :class:`np.ndarray`
    :ivar W: The matrices representing the kernel's weights.
    :vartype W: :class:`np.ndarray`
    :ivar Wpast: The matrices representing the kernel's weights at a
        previous state.
    :vartype Wpast: :class:`np.ndarray`
    :ivar sigma: The influence distance of the kernel.
    :vartype sigma: float
    :ivar name: The name of the layer containing the kernel.
    :vartype name: str
    """
    # ---   INIT   --- #
    # ⨪--------------- #
    def __init__(self, **kwargs):
        """
        Initialize an instance of KPConvLayerPlot.

        :param kwargs: The key-word arguments defining the plot's attributes.
        :type kwargs: dict
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Initialize attributes of KPConvLayerPlot
        self.Q = kwargs.get('Q', None)
        self.W = kwargs.get('W', None)
        self.Wpast = kwargs.get('Wpast', None)
        self.sigma = kwargs.get('sigma', None)
        self.name = kwargs.get('name', None)
        # Validate
        if (
            self.Q is None or len(self.Q) < 1
        ) and (
            self.W is None or len(self.W) < 1
        ):
            raise PlotException(
                'KPConvLayerPlot MUST receive at least the kernel structure '
                '(Q) or the kernel weights (W).'
            )
        if self.Q is not None and len(self.Q) > 0 and self.sigma is None:
            raise PlotException(
                'KPConvLayerPlot MUST receive the influence distance (sigma) '
                'if the structure space is given.'
            )

    # ---   PLOT METHODS   --- #
    # ------------------------ #
    def plot(self, **kwargs):
        """
        Plot the structure space and the matrices of weights representing the
        kernel of the KPConv layer.

        See :meth:`plot.Plot.plot`.
        """
        # Start time measurement
        start = time.perf_counter()
        # Do the plots
        if self.Q is not None:
            self.plot_kernel_structure(**kwargs)
        if self.W is not None:
            self.plot_kernel_weights(
                self.W,
                'W',
                f'"{self.name}" weights',
                **kwargs
            )
            if self.Wpast is not None:
                self.plot_kernel_weights(
                    self.W-self.Wpast,
                    'Wdiff',
                    f'"{self.name}" weights diff.',
                    **kwargs
                )
        # End time measurement
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'KPConvLayerPlot generated and wrote figures to '
            f'"{self.path}" in {end-start:.3f} seconds.'
        )

    def plot_kernel_structure(self, **kwargs):
        """
        Plot the kernel's structure.

        :param kwargs: The key-word arguments:
        :return: Nothing, but the plot is written to a file.
        """
        # Prepare figure
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'"{self.name}" structure')
        # Prepare 3D axes
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # Plot points
        ax.scatter(
            self.Q[:, 0],
            self.Q[:, 1],
            self.Q[:, 2],
            color='red',
            depthshade=False,
            s=64,
            edgecolor='black',
            lw=1.5
        )
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)*self.sigma
        y = np.sin(u)*np.sin(v)*self.sigma
        z = np.cos(v)*self.sigma
        # Plot influence regions
        for q in self.Q:
            ax.plot_wireframe(
                x+q[0], y+q[1], z+q[2],
                color='black'
            )
            ax.plot_surface(
                x+q[0], y+q[1], z+q[2],
                color='red', shade=False, alpha=0.3
            )
        # Make plot effective
        path = self.path
        self.path = path + 'Q.svg'
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
        self.path = path

    def plot_kernel_weights(self, W, plot_name, plot_title, **kwargs):
        """
        Plot the kernel's weights.

        :param W: The tensor whose slices are the weights of the kernel.
        :param plot_name: The name of the plot
        :param plot_title: The title of the plot
        :param kwargs: The key-word arguments.
        :return: Nothing, but the plot is written to a file.
        """
        # Prepare figure
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(plot_title)
        num_matrices = W.shape[0]
        nrows, ncols = PlotUtils.rows_and_cols_for_nplots(num_matrices)
        # Plot weights matrices
        for k, Wk in enumerate(W):
            ax = fig.add_subplot(nrows, ncols, k+1)
            ax.matshow(Wk, origin='lower')
            ax.xaxis.tick_bottom()
        # Make plot effective
        path = self.path
        self.path = path + f'{plot_name}.svg'
        self.save_show_and_clear(out_prefix=kwargs.get('out_prefix', None))
        self.path = path
