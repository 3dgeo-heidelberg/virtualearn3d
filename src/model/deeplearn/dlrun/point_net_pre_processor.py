# ---   IMPORTS   --- #
# ------------------- #
from src.utils.receptive_field import ReceptiveField
import src.main.main_logger as LOGGING
import numpy as np
from scipy.spatial import KDTree as KDT
import scipy.stats


# ---   CLASS   --- #
# ----------------- #
class PointNetPreProcessor:
    r"""
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed into the PointNet neural network.

    :ivar sphere_radius: The radius of the sphere that bounds a neighborhood.
        For an arbitrary set of 3D points it can be found as follows:

        .. math::
            r = \dfrac{1}{2} \, \max \; \biggl\{
                x_{\mathrm{max}}-x_{\mathrm{min}},
                y_{\mathrm{max}}-y_{\mathrm{min}},
                z_{\mathrm{max}}-z_{\mathrm{min}}
            \biggr\}

    :vartype sphere_radius: float
    :ivar separation_factor: How many times the sphere radius separates the
        support points to find the input neighborhoods that will be used to
        generate the receptive fields for the neural network.

        For a given separation factor :math:`k`, the following condition
        should be satisfied to prevent missing any region of the input
        point cloud on a :math:`n`-dimensional space:

        .. math::
            k \leq \dfrac{2}{\sqrt{n}}

    :vartype separation_factor: float
    :ivar cell_size: The cell size defining the receptive field. See
        :class:`ReceptiveField`.
    :vartype cell_size: :class:`np.ndarray`
    :ivar last_call_receptive_fields: List of the receptive fields used the
        last time that the pre-processing logic was executed.
    :vartype last_call_receptive_fields: list
    :ivar last_call_neighborhoods: List of neighborhoods (represented by
        indices) used the last time that the pre-processing logic was
        executed.
    :vartype last_call_neighborhoods: list
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization/instantiation of a PointNet pre-processor.

        :param kwargs: The key-word arguments for the PointNetPreProcessor.
        """
        # Assign attributes
        self.sphere_radius = kwargs.get('sphere_radius', 1.0)
        self.separation_factor = kwargs.get('separation_factor', np.sqrt(3)/4)
        self.cell_size = np.array(kwargs.get('cell_size', [0.1, 0.1, 0.1]))
        # Initialize last call cache
        self.last_call_receptive_fields = None
        self.last_call_neighborhoods = None

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        """
        Executes the pre-processing logic. It also updates the cache-like
        variables of the preprocessor.

        The pre-processing logic consists of steps:

        **1) Generate support points.**

        Support points are generated as a set of points separated in :math:`kr`
        units between them and covering the entire point cloud. Watch out,
        generating support points with
        :math:`k > 2/\sqrt{n}` in a :math:`n`-dimensional
        space leads to gaps between the bounding spheres centered on the
        support points.

        **2) Find neighborhoods centered on support points.**
        For each support point, a neighborhood (by default a spherical
        neighborhood but the neighborhood definition can be arbitrarily
        changed) centered on it is found.

        **3) Filter empty neighborhoods.**
        Any support point that leads to an empty neighborhood when considering
        as neighbors points from the input point cloud is filtered out.

        **4) Trasnform non-empty neighborhoods to receptive fields.**
        Each non-empty neighborhood is transformed to a receptive field.
        See :class:`.ReceptiveField` for further details.

        :param inputs: A key-word input where the key "X" gives the input
            dataset and the "y" (OPTIONALLY) gives the reference values that
            can be used to fit/train a PointNet model.
        :type inputs: dict
        :return: Either (Xout, yout) or Xout. Where Xout are the points
            representing the receptive field and yout (only given when
            "y" was given in the inputs dictionary) the corresponding
            reference values for those points.
        """
        # Extract inputs
        X, y = inputs['X'], inputs.get('y', None)
        # Build support points
        xmin, xmax = np.min(X, axis=0), np.max(X, axis=0)
        l = self.separation_factor * self.sphere_radius  # Cell size
        A, B, C = np.meshgrid(
            np.concatenate([np.arange(xmin[0], xmax[0], l), [xmax[0]]]),
            np.concatenate([np.arange(xmin[1], xmax[1], l), [xmax[1]]]),
            np.concatenate([np.arange(xmin[2], xmax[2], l), [xmax[2]]])
        )
        sup_X = np.array([A.flatten(), B.flatten(), C.flatten()]).T
        # Extract neighborhoods
        # TODO Rethink : Chunk queries to prevent OOM
        kdt = KDT(X)
        kdt_sup = KDT(sup_X)
        I = kdt_sup.query_ball_tree(kdt, self.sphere_radius)  # Neigh. indices
        # Remove empty neighborhoods and corresponding support points
        non_empty_mask = [len(Ii) > 0 for Ii in I]
        I = [Ii for i, Ii in enumerate(I) if non_empty_mask[i]]
        sup_X = sup_X[non_empty_mask]
        self.last_call_neighborhoods = [Ii for Ii in I]  # TODO Restore or Rethink
        # Prepare receptive field
        self.last_call_receptive_fields = [
            ReceptiveField(
                bounding_radii=np.array([
                    self.sphere_radius for i in range(X.shape[1])
                ]),
                cell_size=self.cell_size
            ).fit(X[Ii], sup_X[i])
            for i, Ii in enumerate(I)
        ]
        # Neighborhoods ready to be fed into the neural network
        # TODO Rethink : Use support points to build the input ?
        Xout = np.array([
            self.last_call_receptive_fields[i].centroids_from_points(
                X[Ii], interpolate=True
            )
            for i, Ii in enumerate(I)
        ])
        LOGGING.LOGGER.info(
            f'The point net pre processor generated {Xout.shape[0]} receptive '
            'fields. '
        )
        if y is not None:
            yout = np.array([
                self.last_call_receptive_fields[i].reduce_values(
                    Xout[i],
                    y[Ii],
                    reduce_f=lambda x: scipy.stats.mode(x)[0][0],
                    fill_nan=True
                ) for i, Ii in enumerate(I)
            ])
            return Xout, yout
        return Xout
