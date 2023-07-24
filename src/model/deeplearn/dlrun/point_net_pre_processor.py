import numpy as np
from scipy.spatial import KDTree as KDT


class PointNetPreProcessor:
    """
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed into the PointNet neural network.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, num_points, **kwargs):
        self.num_points = num_points
        self.sphere_radius = kwargs.get('sphere_radius', 3.0)
        self.separation_factor = kwargs.get('separation_factor', np.sqrt(3)/4)

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
        kdt = KDT(X)
        kdt_sup = KDT(sup_X)
        I = kdt_sup.query_ball_tree(kdt, self.sphere_radius)  # Neigh. indices
        # Remove empty neighborhoods
        I = [Ii for Ii in I if len(Ii) > 0]
        # Neighborhoods ready to be fed into the neural network
        # TODO Rethink : Use support points to build the input ?
        #Xout = np.expand_dims(X[:self.num_points], 0)  # TODO Rethink : Old
        Xout = [X[Ii] for Ii in I]  # TODO Rethink : New
        if y is not None:
            #yout = np.expand_dims(y[:self.num_points], 0)  # TODO Rethink : Old
            yout = [y[Ii] for Ii in I]  # TODO Rethink : New
            return Xout, yout
        return Xout
