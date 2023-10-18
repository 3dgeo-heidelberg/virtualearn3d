# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.utils.ptransf.receptive_field_gs import ReceptiveFieldGS
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
        status = status and self.test_receptive_field(
            bounding_radii=np.array([1.0, 1.0]),
            cell_size=np.array([1/3, 2/3]),
            center_point=np.array([0.0, 0.0]),
            input_points=np.array([
                [0.79, -0.95],
                [0.38, 0.02],
                [0.57, -0.3],
                [0.37, 0.01],
                [0.81, 0.03],
                [0.77, -0.23],
                [0.11, 0.98],
                [0.12, 0.52],
                [0.17, 0.53],
                [0.17, 0.74],
                [0.63, 0.42],
                [0.59, 0.71],
                [0.54, 0.38],
                [0.92, 0.81],
                [0.80, 0.48]
            ]),
            values_to_propagate=np.array([
                [0, 5],
                [1, 10],
                [2, 11],
                [3, 15],
                [4, 16],
                [5, 17]
            ], dtype=int),
            expected_num_cells=18,
            expected_centroid_nointerp=np.array([
                [np.nan, np.nan],           # mu_0
                [np.nan, np.nan],           # mu_1
                [np.nan, np.nan],           # mu_2
                [np.nan, np.nan],           # mu_3
                [np.nan, np.nan],           # mu_4
                [0.79, -0.95],              # mu_5
                [np.nan, np.nan],           # mu_6
                [np.nan, np.nan],           # mu_7
                [np.nan, np.nan],           # mu_8
                [np.nan, np.nan],           # mu_9
                [0.44, -0.09],              # mu_10
                [0.79, -0.1],               # mu_11
                [np.nan, np.nan],           # mu_12
                [np.nan, np.nan],           # mu_13
                [np.nan, np.nan],           # mu_14
                [0.1425, 0.6925],           # mu_15
                [0.58666666, 0.50333333],   # mu_16
                [0.86, 0.645],              # mu_17
            ]),
            expected_centroid_interp=np.array([
                [0.60152778, 0.11680556],   # mu_0
                [0.60152778, 0.11680556],   # mu_1
                [0.60152778, 0.11680556],   # mu_2
                [0.60152778, 0.11680556],   # mu_3
                [0.60152778, 0.11680556],   # mu_4
                [0.79, -0.95],              # mu_5
                [0.60152778, 0.11680556],   # mu_6
                [0.60152778, 0.11680556],   # mu_7
                [0.60152778, 0.11680556],   # mu_8
                [0.60152778, 0.11680556],   # mu_9
                [0.44, -0.09],              # mu_10
                [0.79, -0.1],               # mu_11
                [0.60152778, 0.11680556],   # mu_12
                [0.60152778, 0.11680556],   # mu_13
                [0.60152778, 0.11680556],   # mu_14
                [0.1425, 0.6925],           # mu_15
                [0.58666666, 0.50333333],   # mu_16
                [0.86, 0.645],              # mu_17
            ]),
            expected_propagated_values=np.array([
                [0, 5],
                *[[1, 10]]*3,
                *[[2, 11]]*2,
                *[[3, 15]]*4,
                *[[4, 16]]*3,
                *[[5, 17]]*2
            ], dtype=int)
        )
        # 3) Test regular 3D receptive field
        status = status and self.test_receptive_field(
            bounding_radii=np.array([1.5, 1.5, 1.5]),
            cell_size=np.array([2/3, 2/3, 2/3]),
            center_point=np.array([1.5, 1.5, 1.5]),
            input_points=np.array([
                [0, 0, 0],
                [1.1, 0.1, 0.1],
                [3.0, 0.2, 0.3],
                [0, 1, 0],
                [1.2, 1.3, 0.1],
                [1.5, 1.6, 0.4],
                [2.5, 1.9, 0.2],
                [0.1, 2.2, 0],
                [1.2, 2.2, 0.1],
                [2.2, 2.2, 0.3],
                [2.2, 2.2, 0.9],
                [0.9, 0.9, 1.9],
                [0.8, 0.9, 1.7],
                [1.8, 0.9, 1.9],
                [2.7, 0.8, 1.8],
                [0.9, 1.1, 1.1],
                [1.9, 1.3, 1.1],
                [2.8, 1.6, 1.2],
                [0.7, 2.3, 1.2],
                [0.7, 2.5, 1.3],
                [1.3, 2.6, 1.4],
                [2.5, 2.9, 1.8],
                [0.9, 0.9, 2.0],
                [1.5, 0.9, 2.4],
                [1.5, 0.8, 2.7],
                [2.1, 0.9, 2.8],
                [0.3, 1.1, 2.0],
                [1.6, 1.3, 2.5],
                [2.9, 1.5, 2.7],
                [0.5, 2.2, 2.5],
                [0.6, 2.4, 2.6],
                [1.7, 2.8, 2.9],
                [3, 3, 3]
            ]),
            values_to_propagate=np.array([i for i in range(27)], dtype=int),
            expected_num_cells=27,
            expected_centroid_nointerp=np.array([
                [0, 0, 0],                  # mu_0
                [1.1, 0.1, 0.1],            # mu_1
                [3.0, 0.2, 0.3],            # mu_2
                [0, 1, 0],                  # mu_3
                [1.35, 1.45, 0.25],         # mu_4
                [2.5, 1.9, 0.2],            # mu_5
                [0.1, 2.2, 0],              # mu_6
                [1.2, 2.2, 0.1],            # mu_7
                [2.2, 2.2, 0.6],            # mu_8
                [0.85, 0.9, 1.8],           # mu_9
                [1.8, 0.9, 1.9],            # mu_10
                [2.7, 0.8, 1.8],            # mu_11
                [0.9, 1.1, 1.1],            # mu_12
                [1.9, 1.3, 1.1],            # mu_13
                [2.8, 1.6, 1.2],            # mu_14
                [0.7, 2.4, 1.25],           # mu_15
                [1.3, 2.6, 1.4],            # mu_16
                [2.5, 2.9, 1.8],            # mu_17
                [0.9, 0.9, 2.0],            # mu_18
                [1.5, 0.85, 2.55],          # mu_19
                [2.1, 0.9, 2.8],            # mu_20
                [0.3, 1.1, 2.0],            # mu_21
                [1.6, 1.3, 2.5],            # mu_22
                [2.9, 1.5, 2.7],            # mu_23
                [0.55, 2.3, 2.55],          # mu_24
                [1.7, 2.8, 2.9],            # mu_25
                [3, 3, 3]                   # mu_26
            ]),
            expected_centroid_interp=np.array([
                [0, 0, 0],                  # mu_0
                [1.1, 0.1, 0.1],            # mu_1
                [3.0, 0.2, 0.3],            # mu_2
                [0, 1, 0],                  # mu_3
                [1.35, 1.45, 0.25],         # mu_4
                [2.5, 1.9, 0.2],            # mu_5
                [0.1, 2.2, 0],              # mu_6
                [1.2, 2.2, 0.1],            # mu_7
                [2.2, 2.2, 0.6],            # mu_8
                [0.85, 0.9, 1.8],           # mu_9
                [1.8, 0.9, 1.9],            # mu_10
                [2.7, 0.8, 1.8],            # mu_11
                [0.9, 1.1, 1.1],            # mu_12
                [1.9, 1.3, 1.1],            # mu_13
                [2.8, 1.6, 1.2],            # mu_14
                [0.7, 2.4, 1.25],           # mu_15
                [1.3, 2.6, 1.4],            # mu_16
                [2.5, 2.9, 1.8],            # mu_17
                [0.9, 0.9, 2.0],            # mu_18
                [1.5, 0.85, 2.55],          # mu_19
                [2.1, 0.9, 2.8],            # mu_20
                [0.3, 1.1, 2.0],            # mu_21
                [1.6, 1.3, 2.5],            # mu_22
                [2.9, 1.5, 2.7],            # mu_23
                [0.55, 2.3, 2.55],          # mu_24
                [1.7, 2.8, 2.9],            # mu_25
                [3, 3, 3]                   # mu_26
            ]),
            expected_propagated_values=np.array([
                0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8,
                9, 9, 10, 11, 12, 13, 14, 15, 15, 16, 17,
                18, 19, 19, 20, 21, 22, 23, 24, 24, 25, 26
            ], dtype=int)
        )
        # 4) Test irregular 3D receptive field
        status = status and self.test_receptive_field(
            bounding_radii=np.array([7.5, 3.5, 5]),
            cell_size=np.array([2/3, 2, 1]),
            center_point=np.array([7.5, 3.5, 5]),
            input_points=np.array([
                [0.1, 1, 0.5],
                [4, 3, 2],
                [5.5, 7, 0],
                [15, 0, 0],
                [12, 6, 4],
                [7, 7, 7],
                [8, 8, 8],
                [11, 7, 5]
            ]),
            values_to_propagate=np.array(
                [[np.pi/i, np.exp(i)] for i in range(6, 0, -1) if i !=3],
                dtype=float
            ),
            expected_num_cells=6,
            expected_centroid_nointerp=np.array([
                [2.05, 2, 1.25],            # mu_0
                [5.5, 7, 0],                # mu_1
                [13.5, 3, 2],               # mu_2
                [np.nan, np.nan, np.nan],   # mu_3
                [7.5, 7.5, 7.5],            # mu_4
                [11, 7, 5]                  # mu_5
            ]),
            expected_centroid_interp=np.array([
                [2.05, 2, 1.25],            # mu_0
                [5.5, 7, 0],                # mu_1
                [13.5, 3, 2],               # mu_2
                [7.91, 5.3, 3.15],          # mu_3
                [7.5, 7.5, 7.5],            # mu_4
                [11, 7, 5]                  # mu_5
            ]),
            expected_propagated_values=np.array([
                [np.pi/6, np.exp(6)],
                [np.pi/6, np.exp(6)],
                [np.pi/5, np.exp(5)],
                [np.pi/4, np.exp(4)],
                [np.pi/4, np.exp(4)],
                [np.pi/2, np.exp(2)],
                [np.pi/2, np.exp(2)],
                [np.pi/1, np.exp(1)]
            ], dtype=float)
        )
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
        rf = ReceptiveFieldGS(
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
        sum_abs_diff = np.sum(np.abs(centroid_interp-expected_centroid_interp))
        if sum_abs_diff > eps or np.any(np.isnan(sum_abs_diff)):
            return False
        # Validate propagated values
        propagated_values = rf.propagate_values(values_to_propagate, safe=True)
        if np.sum(np.abs(propagated_values-expected_propagated_values)) > eps:
            return False
        # Return on success
        return True
