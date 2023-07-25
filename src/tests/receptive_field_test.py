# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest, VL3DTestException
from src.utils.receptive_field import ReceptiveField
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class ReceptiveFieldTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Receptive field test that checks the indexing matrix, the centroids, and
    the propagation work as expected for some use cases.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Receptive field test')

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run receptive field test.

        :return: True if ReceptiveField works as expected for the test cases,
            False otherwise.
        :rtype: bool
        """
        status = True
        # 1) Test regular 2D receptive field
        status = status and self.test_receptive_field(
            bounding_radii=np.array([5.0, 5.0]),
            cell_size=np.array([0.4, 0.4]),
            center_point=np.array([5.0, 5.0]),
            input_points=np.array([
                [7.5, 1.7],
                [8.8, 1.1],
                [8.7, 0.5],
                [5.2, 2.7],
                [4.4, 2.8],
                [7.6, 2.6],
                [7.1, 2.6],
                [9.8, 2.3],
                [9.5, 2.8],
                [9.4, 3.4],
                [5.1, 5.7],
                [4.6, 4.2],
                [7.6, 5.2],
                [7, 5],
                [6.4, 5],
                [9.4, 5.7],
                [9.3, 5.2],
                [8.4, 4.3],
                [8.7, 4.7],
                [7.1, 6.6],
                [6.0, 6.2],
                [6.1, 7.8],
                [9, 7.2],
                [9.9, 7],
                [9.1, 9.8]
            ]),
            values_to_propagate=np.array([
                [0, 3],
                [1, 4],
                [2, 7],
                [3, 8],
                [4, 9],
                [5, 12],
                [6, 13],
                [7, 14],
                [8, 18],
                [9, 19],
                [10, 24]
            ], dtype=int),
            expected_num_cells=25,
            expected_centroid_nointerp=np.array([
                [np.nan, np.nan],           # mu_0
                [np.nan, np.nan],           # mu_1
                [np.nan, np.nan],           # mu_2
                [7.5, 1.7],                 # mu_3
                [8.75, 0.8],                # mu_4
                [np.nan, np.nan],           # mu_5
                [np.nan, np.nan],           # mu_6
                [4.8, 2.75],                # mu_7
                [7.35, 2.6],                # mu_8
                [9.5666666, 2.8333333],     # mu_9
                [np.nan, np.nan],           # mu_10
                [np.nan, np.nan],           # mu_11
                [4.85, 4.95],               # mu_12
                [7, 5.0666666],             # mu_13
                [8.95, 4.975],              # mu_14
                [np.nan, np.nan],           # mu_15
                [np.nan, np.nan],           # mu_16
                [np.nan, np.nan],           # mu_17
                [6.4, 6.8666666],           # mu_18
                [9.45, 7.1],                # mu_19
                [np.nan, np.nan],           # mu_20
                [np.nan, np.nan],           # mu_21
                [np.nan, np.nan],           # mu_22
                [np.nan, np.nan],           # mu_23
                [9.1, 9.8]                  # mu_24
            ]),
            expected_centroid_interp=np.array([
                [7.02708332, 3.44583331],   # mu_0
                [7.02708332, 3.44583331],   # mu_1
                [7.34583332, 3.20937499],   # mu_2
                [7.5, 1.7],                 # mu_3
                [8.75, 0.8],                # mu_4
                [6.95, 3.71354165],         # mu_5
                [6.95, 3.71354165],         # mu_6
                [4.8, 2.75],                # mu_7
                [7.35, 2.6],                # mu_8
                [9.5666666, 2.8333333],     # mu_9
                [6.95, 3.71354165],         # mu_10
                [6.95, 3.71354165],         # mu_11
                [4.85, 4.95],               # mu_12
                [7, 5.0666666],             # mu_13
                [8.95, 4.975],              # mu_14
                [7.0375, 4.50104165],       # mu_15
                [7.0375, 4.50104165],       # mu_16
                [7.0375, 4.50104165],       # mu_17
                [6.4, 6.8666666],           # mu_18
                [9.45, 7.1],                # mu_19
                [7.2375, 5.51354165],       # mu_20
                [7.2375, 5.51354165],       # mu_21
                [7.2375, 5.51354165],       # mu_22
                [7.2375, 5.51354165],       # mu_23
                [9.1, 9.8]                  # mu_24
            ]),
            expected_propagated_values=np.array([
                [0, 3],
                *[[1, 4]]*2,
                *[[2, 7]]*2,
                *[[3, 8]]*2,
                *[[4, 9]]*3,
                *[[5, 12]]*2,
                *[[6, 13]]*3,
                *[[7, 14]]*4,
                *[[8, 18]]*3,
                *[[9, 19]]*2,
                [10, 24]
            ], dtype=int)
        )
        # 2) Test irregular 2D receptive field
        # TODO Rethink : Implement
        # 3) Test regular 3D receptive field
        # TODO Rethink : Implement
        # 4) Test irregular 3D receptive field
        # TODO Rethink : Implement
        # Return
        return status

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    @staticmethod
    def test_receptive_field(
        bounding_radii,
        cell_size,
        center_point,
        input_points,
        values_to_propagate,
        expected_num_cells,
        expected_centroid_nointerp,
        expected_centroid_interp,
        expected_propagated_values,
        eps=1e-6
    ):
        """
        Build a receptive field with the input points and propagate the given
        values. Then, the results are compared with the expected outputs to
        validate the implementation.

        :param bounding_radii: The bounding radii to build the receptive
            field.
        :type bounding_radii: :float:
        :param cell_size: The cell size to build the receptive field.
        :type cell_size: :class:`np.ndarray`
        :param center_point: The center point for the receptive field.
        :type center_point: :class:`np.ndarray`
        :param input_points: The input points to build the receptive field.
        :type input points: :class:`np.ndarray`
        :param values_to_propagate: The values to propagate through the
            receptive field.
        :type values_to_propagate: :class:`np.ndarray`
        :param expected_num_cells: The expected number of cells composing the
            receptive field.
        :type expected_num_cells: int
        :param expected_centroid_nointerp: The expected centroids (without
            interpolation).
        :type expected_centroid_nointerp: :class:`np.ndarray`
        :param expected_centroid_interp: The expected centroids (with
            interpolation).
        :type expected_centroid_interp: :class:`np.ndarray`
        :param expected_propagated_values: The expected propagated values.
        :type expected_propagated_values: :class:`np.ndarray`
        :param eps: The decimal tolerance for numerical validation.
        :type eps: float
        :return: True if the receptive field behaves as expected, False
            otherwise.
        """
        # Instantiate and fit receptive field
        rf = ReceptiveField(
            cell_size=cell_size,
            bounding_radii=bounding_radii
        )
        rf.fit(input_points, center_point)
        if rf.num_cells != expected_num_cells:
            return False
        # Validate centroids with no interp
        centroid_nointerp = rf.undo_center_and_scale(rf.centroids_from_points(
            input_points,
            interpolate=False
        ))
        centroid_nointerp = centroid_nointerp[
            ~np.sum(np.isnan(centroid_nointerp), axis=1, dtype=bool)
        ]
        expected_centroid_nointerp = expected_centroid_nointerp[
            ~np.sum(np.isnan(expected_centroid_nointerp), axis=1, dtype=bool)
        ]
        if len(centroid_nointerp) != len(expected_centroid_nointerp):
            return False
        if np.sum(np.abs(centroid_nointerp-expected_centroid_nointerp)) > eps:
            return False
        # Validate centroid with interp
        centroid_interp = rf.undo_center_and_scale(rf.centroids_from_points(
            input_points,
            interpolate=True
        ))
        if np.sum(np.abs(centroid_interp-expected_centroid_interp)) > eps:
            return False
        # Validate propagated values
        propagated_values = rf.propagate_values(values_to_propagate, safe=True)
        if np.sum(np.abs(propagated_values-expected_propagated_values)) > eps:
            return False
        # Return on success
        return True
