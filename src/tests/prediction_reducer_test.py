# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.utils.preds.prediction_reducer import PredictionReducer
from src.utils.preds.mean_pred_reduce_strategy import MeanPredReduceStrategy
from src.utils.preds.sum_pred_reduce_strategy import SumPredReduceStrategy
from src.utils.preds.max_pred_reduce_strategy import MaxPredReduceStrategy
from src.utils.preds.entropic_pred_reduce_strategy \
    import EntropicPredReduceStrategy
from src.utils.preds.argmax_pred_select_strategy import ArgMaxPredSelectStrategy
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class PredictionReducerTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Prediction reducer test that checks the different strategies work
    correctly.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Prediction reducer test')
        self.eps = 1e-5  # Decimal tolerance error

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run prediction reducer test

        :return: True if PredictionReducer works as expected for the test
            cases, False otherwise.
        :rtype: bool
        """
        # Build test case : Single value
        num_points_single = 7
        Z_single = [
            np.array([0.31, 0.12, 0.56, 0.94, 0.78, 0.23, 0.65]),
            np.array([0.22, 0.96, 0.84, 0.48, 0.63, 0.75]),
            np.array([0.57, 0.84, 0.76, 0.65, 0.61]),
            np.array([0.23, 0.66, 0.23, 0.91, 0.68, 0.33, 0.29]),
            np.array([0.91, 0.21, 0.37])
        ]
        I_single = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [5, 1, 0, 2, 4],
            [0, 2, 4, 6, 1, 3, 5],
            [0, 2, 6],
        ]
        Z_single_sum_argmax_ref = np.array([
            2.21, 1.86, 3.04, 2.11, 2.1, 1.72, 2.68
        ])
        Z_single_mean_argmax_ref = np.array([
            0.5525, 0.465, 0.608, 0.70333333, 0.525, 0.43, 0.67
        ])
        Z_single_max_argmax_ref = np.array([
            0.91, 0.84, 0.96, 0.94, 0.78, 0.63, 0.91
        ])
        Z_single_entro_argmax_ref = np.array([
            0.81108951, 0.58721503, 0.78694629, 0.89615775, 0.66675506,
            0.52072115, 0.81960945
        ])
        Y_single_sum_argmax_ref = np.array([2, 2, 3, 2, 2, 2, 3])
        Y_single_mean_argmax_ref = np.array([1, 0, 1, 1, 1, 0, 1])
        Y_single_max_argmax_ref = np.array([1, 1, 1, 1, 1, 1, 1])
        Y_single_entro_argmax_ref = np.array([1, 1, 1, 1, 1, 1, 1])

        # Build test case : Many values
        num_points_many, num_vals_many = 7, 5
        Z_many = [
            np.array([
                [0.1,   0.1,    0.2,    0.15,   0.45],
                [0,     0,      0.3,    0,      0.7 ],
                [0.5,   0.11,   0.11,   0.15,   0.13]
            ]),
            np.array([
                [0.05,  0.1,    0.05,   0.75,   0.05]
            ]),
            np.array([
                [0.1,   0.05,   0.2,    0.03,   0.62],
                [0.2,   0,      0.7,    0.01,   0.09]
            ]),
            np.array([
                [0.61, 0.23,    0.11,   0,      0.05]
            ]),
            np.array([
                [0.13,  0.5,    0.15,   0.11,   0.11],
                [0.2,   0.0,    0.0,    0.15,   0.65],
                [0.08,   0.02,   0.80,   0.03,  0.07],
                [0.9,   0,      0.01,   0.06,   0.03]
            ]),
            np.array([
                [0.11,  0.11,    0.5,   0.13,   0.15],
                [0.0,   0.0,     1.0,    0.0,   0.0 ],
                [0.07,  0.03,   0.78,   0.03,   0.09],
                [0.89,  0.09,   0.01,    0.0,   0.01],
                [0.1,   0.65,    0.1,   0.05,   0.1]
            ]),
            np.array([
                [0.03,  0.02,   0.09,    0.11,  0.75],
                [0.67,  0.05,   0.07,    0.18,  0.03]
            ])
        ]
        I_many = [
            [0, 1, 2],
            [3],
            [4, 5],
            [6],
            [0, 1, 2, 4],
            [5, 6, 0, 1, 2],
            [4, 5]
        ]
        Z_many_sum_argmax_ref = np.array([
            [0.3,  0.63, 1.13, 0.29, 0.65],
            [1.09, 0.09, 0.31, 0.15, 1.36],
            [0.68, 0.78, 1.01, 0.23, 0.3 ],
            [0.05, 0.1 , 0.05, 0.75, 0.05],
            [1.03, 0.07, 0.3,  0.2,  1.4 ],
            [0.98, 0.16, 1.27, 0.32, 0.27],
            [0.61, 0.23, 1.11, 0.0,   0.05]
        ])
        Z_many_mean_argmax_ref = np.array([
            [0.1,        0.21,       0.37666667, 0.09666667, 0.21666667],
            [0.36333333, 0.03,       0.10333333, 0.05,       0.45333333],
            [0.22666667, 0.26,       0.33666667, 0.07666667, 0.1       ],
            [0.05,       0.1,        0.05,       0.75,       0.05      ],
            [0.34333333, 0.02333333, 0.1,        0.06666667, 0.46666667],
            [0.32666667, 0.05333333, 0.42333333, 0.10666667, 0.09      ],
            [0.305,      0.115,      0.555,      0,          0.025     ]
        ])
        Z_many_max_argmax_ref = np.array([
            [0.13, 0.5,  0.78, 0.15, 0.45],
            [0.89, 0.09, 0.3,  0.15, 0.7 ],
            [0.5,  0.65, 0.8,  0.15, 0.13],
            [0.05, 0.1,  0.05, 0.75, 0.05],
            [0.9,  0.05, 0.2,  0.11, 0.75],
            [0.67, 0.11, 0.7,  0.18, 0.15],
            [0.61, 0.23, 1.0,   0.0, 0.05]
        ])
        Z_many_entro_argmax_ref = np.array([
            [9.09360670e-02, 1.58179030e-01, 5.02641948e-01, 7.52821067e-02, 1.72960848e-01],
            [4.04761321e-01, 3.55903256e-02, 1.06094998e-01, 3.96143740e-02, 4.13940926e-01],
            [1.71148276e-01, 2.38066991e-01, 4.38576160e-01, 6.05751321e-02, 9.16334406e-02],
            [5.00000000e-02, 1.00000000e-01, 5.00000000e-02, 7.50000000e-01, 5.00000000e-02],
            [4.40952388e-01, 1.79586204e-02, 7.94898848e-02, 6.85822824e-02, 3.93017278e-01],
            [3.50993712e-01, 4.00137550e-02, 4.33381963e-01, 9.51703705e-02, 8.04406395e-02],
            [1.86059838e-01, 7.01541425e-02, 7.28536660e-01, 1.00000000e-06, 1.52514444e-02]
        ])
        Y_many_sum_argmax_ref = np.array([2, 4, 2, 3, 4, 2, 2])
        Y_many_mean_argmax_ref = np.array([2, 4, 2, 3, 4, 2, 2])
        Y_many_max_argmax_ref = np.array([[2, 0, 2, 3, 0, 2, 2]])
        Y_many_entro_argmax_ref = np.array([2, 4, 2, 3, 0, 2, 2])

        # Prepare prediction reducers
        pr_sum_argmax = PredictionReducer(
            reduce_strategy=SumPredReduceStrategy(),
            select_strategy=ArgMaxPredSelectStrategy()
        )
        pr_mean_argmax = PredictionReducer(
            reduce_strategy=MeanPredReduceStrategy(),
            select_strategy=ArgMaxPredSelectStrategy()
        )
        pr_max_argmax = PredictionReducer(
            reduce_strategy=MaxPredReduceStrategy(),
            select_strategy=ArgMaxPredSelectStrategy()
        )
        pr_entro_argmax = PredictionReducer(
            reduce_strategy=EntropicPredReduceStrategy(),
            select_strategy=ArgMaxPredSelectStrategy()
        )

        # Test prediction reducers on single
        if not self.check_pr(
            pr_sum_argmax, num_points_single, 1, Z_single, I_single,
            Z_single_sum_argmax_ref, Y_single_sum_argmax_ref
        ):
            return False
        if not self.check_pr(
            pr_mean_argmax, num_points_single, 1, Z_single, I_single,
            Z_single_mean_argmax_ref, Y_single_mean_argmax_ref
        ):
            return False
        if not self.check_pr(
            pr_max_argmax, num_points_single, 1, Z_single, I_single,
            Z_single_max_argmax_ref, Y_single_max_argmax_ref
        ):
            return False
        if not self.check_pr(
            pr_entro_argmax, num_points_single, 1, Z_single, I_single,
            Z_single_entro_argmax_ref, Y_single_entro_argmax_ref
        ):
            return False

        # Test prediction reducers on many
        if not self.check_pr(
            pr_sum_argmax, num_points_many, num_vals_many, Z_many, I_many,
            Z_many_sum_argmax_ref, Y_many_sum_argmax_ref
        ):
            return False
        if not self.check_pr(
            pr_mean_argmax, num_points_many, num_vals_many, Z_many, I_many,
            Z_many_mean_argmax_ref, Y_many_mean_argmax_ref
        ):
            return False
        if not self.check_pr(
            pr_max_argmax, num_points_many, num_vals_many, Z_many, I_many,
            Z_many_max_argmax_ref, Y_many_max_argmax_ref
        ):
            return False
        if not self.check_pr(
            pr_entro_argmax, num_points_many, num_vals_many, Z_many, I_many,
            Z_many_entro_argmax_ref, Y_many_entro_argmax_ref
        ):
            return False
        # All tests passed
        return True

    # ---   CHECK METHODS   --- #
    # ------------------------- #
    def check_pr(self, pr, num_points, num_vals, Z, I, Z_ref, Y_ref):
        """
        Check that the prediction reducer yields the expected output.

        :param pr: The prediction reducer itself.
        :param num_points: The number of output points.
        :param num_vals: The number of point-wise variables.
        :param Z: The input values.
        :param I: The input neighborhoods.
        :param Z_ref: The expected reduced values.
        :param Y_ref: The expected selected values.
        :return: True if the output is as expected, False otherwise.
        """
        # Check reduce
        Z_red = pr.reduce(num_points, num_vals, Z, I)
        if not np.allclose(Z_red, Z_ref, atol=self.eps):
            return False
        # Check select
        Y_sel = pr.select(Z_red)
        if not np.allclose(Y_sel, Y_ref, atol=self.eps):
            return False
        # Check passed
        return True
