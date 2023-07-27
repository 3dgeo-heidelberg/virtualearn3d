from src.utils.receptive_field import ReceptiveField
import numpy as np
from scipy.spatial import KDTree as KDT
import scipy.stats


class PointNetPreProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed into the PointNet neural network.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        # Assign attributes
        self.sphere_radius = kwargs.get('sphere_radius', 1.0)
        self.separation_factor = kwargs.get('separation_factor', np.sqrt(3)/4)
        self.separation_factor = 2.0  # TODO Remove
        self.cell_size = np.array(kwargs.get('cell_size', [0.1, 0.1, 0.1]))
        # Initialize last call cache
        self.last_call_receptive_fields = None

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
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
        if y is not None:
            yout = np.array([
                self.last_call_receptive_fields[i].reduce_values(
                    Xout[i],
                    y[Ii],
                    reduce_f=lambda x: scipy.stats.mode(x)[0][0],
                    fill_nan=True
                ) for i, Ii in enumerate(I)
            ])  # TODO Rethink : New
            print(f'Num training cases: {len(yout)}')  # TODO Remove
            return Xout, yout
        return Xout
