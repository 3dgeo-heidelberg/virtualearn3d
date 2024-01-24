# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.utils.ptransf.receptive_field_hierarchical_fps import \
    ReceptiveFieldHierarchicalFPS
from src.utils.ptransf.receptive_field_fps import \
    ReceptiveFieldFPS
from scipy.spatial import KDTree as KDT
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class HierarchicalReceptiveFieldTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Hierarchical receptive field test that checks the receptive field at
    each depth of the hierarchy.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Hierarchical receptive field test')

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run receptive field test.

        :return: True if ReceptiveField works as expected for the test cases,
            False otherwise.
        :rtype: bool
        """
        # Generate test data
        angular_steps = 256
        num_points = angular_steps**2
        theta, phi = [np.linspace(0, 2*np.pi, angular_steps)]*2
        R, r = 1.0, 0.5  # Torus radii
        X = np.array([
            [  # x coordinates
                (R+r*np.cos(theta_i))*np.cos(phi_i)
                for theta_i in theta for phi_i in phi
            ],
            [  # y coordinates
                (R+r*np.cos(theta_i))*np.sin(phi_i)
                for theta_i in theta for phi_i in phi
            ],
            [  # z coordinates
                r*np.sin(theta_i)
                for theta_i in theta for phi_i in phi
            ]
        ]).T
        F = np.random.normal(0, 1, (num_points, 1))
        # Instantiate hierarchical FPS receptive field
        num_points_per_depth = [1024, 512, 256, 128, 64]
        num_downsampling_neighbors = [1, 16, 8, 8, 4]
        num_pwise_neighbors = [64, 32, 32, 16, 8]
        num_upsampling_neighbors = [1, 16, 8, 8, 4]
        hrf = ReceptiveFieldHierarchicalFPS(
            num_points_per_depth=num_points_per_depth,
            fast_flag_per_depth=[False]*len(num_points_per_depth),
            num_downsampling_neighbors=num_downsampling_neighbors,
            num_pwise_neighbors=num_pwise_neighbors,
            num_upsampling_neighbors=num_upsampling_neighbors
        )
        # Generate the many receptive fields in a hierarchical way
        mu_X = np.mean(X, axis=0)
        hrf.fit(X, mu_X)
        # Validation thresholds
        val_max_knn_dist = [0.12, 0.16, 0.23, 0.32, 0.45]
        # Validate the depth
        if hrf.max_depth != len(num_points_per_depth):
            return False
        # Validate the receptive field at each depth in the hierarchy
        for d in range(hrf.max_depth):
            # Validate centroids
            Yd = hrf.Ys[d]
            if Yd.shape[0] != num_points_per_depth[d]:
                return False
            kdt = KDT(Yd)
            knn_dist = kdt.query(hrf.center_and_scale(X), k=1)[0]
            max_knn_dist = np.max(knn_dist)
            if max_knn_dist > val_max_knn_dist[d]:
                return False
            # Validate neighborhood cardinalities
            if hrf.NDs[d].shape[-1] != num_downsampling_neighbors[d]:
                return False
            if hrf.NUs[d].shape[-1] != num_upsampling_neighbors[d]:
                return False
            if hrf.Ns[d].shape[-1] != num_pwise_neighbors[d]:
                return False
            # Validate neighborhoods are truly knn
            Xd = hrf.center_and_scale(X) if d == 0 else hrf.Ys[d-1]
            NDd, Nd, NUd = hrf.NDs[d], hrf.Ns[d], hrf.NUs[d]
            for i in range(0, Yd.shape[0], int(np.ceil(Yd.shape[0]/100))):
                # Validate knn for downsampling
                max_knn_dist = np.linalg.norm(Xd[NDd[i]] - Yd[i], axis=1)
                sq_dist = np.sum(np.power(Yd[i]-Xd, 2), axis=1)
                ref_sq_dist = sq_dist[
                    np.argsort(sq_dist)[num_downsampling_neighbors[d]-1]
                ]
                ref_dist = np.sqrt(ref_sq_dist)
                if np.any(max_knn_dist > ref_dist):
                    return False
                # Validate knn for pwise neighborhoods
                max_knn_dist = np.linalg.norm(Yd[Nd[i]] - Yd[i], axis=1)
                sq_dist = np.sum(np.power(Yd[i]-Yd, 2), axis=1)
                ref_sq_dist = sq_dist[
                    np.argsort(sq_dist)[num_pwise_neighbors[d]-1]
                ]
                ref_dist = np.sqrt(ref_sq_dist)
                if np.any(max_knn_dist > ref_dist):
                    return False
                # Validate knn for upsampling
                max_knn_dist = np.linalg.norm(Yd[NUd[i]] - Xd[i], axis=1)
                sq_dist = np.sum(np.power(Xd[i]-Yd, 2), axis=1)
                ref_sq_dist = sq_dist[
                    np.argsort(sq_dist)[num_upsampling_neighbors[d]-1]
                ]
                ref_dist = np.sqrt(ref_sq_dist)
                if np.any(max_knn_dist > ref_dist):
                    return False
        # Generate simple FPS receptive field
        rf = ReceptiveFieldFPS(
            num_points=num_points_per_depth[0],
            num_encoding_neighbors=num_downsampling_neighbors[0],
            fast=False
        )
        rf.fit(X, mu_X)
        # Validate centroids against simple FPS receptive field
        if not np.allclose(
            rf.centroids_from_points(None),
            hrf.centroids_from_points(None)
        ):
            return False
        # Validate reductions against simple FPS receptive field
        reduced_f0 = rf.reduce_values(None, F[:, 0])
        if not np.allclose(reduced_f0, hrf.reduce_values(None, F[:, 0])):
            return False
        # Validate propagations against simple FPS receptive field
        if not np.allclose(
            rf.propagate_values(reduced_f0),
            hrf.propagate_values(reduced_f0)
        ):
            return False
        # Return
        return True
