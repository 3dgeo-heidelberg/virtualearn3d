import pytest
import numpy as np
from src.eval.raster_grid_evaluator import RasterGridEvaluator
from src.main.main_logger import main_logger_init
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade

# ---   JSON SPEC   --- #
# --------------------- #
SPEC = {
    "xres": 1.0,
    "yres": 1.0,
    "radius_expr": "sqrt(2)*l/2",
    "grids": [
        {
            "fnames": ["f1"],
            "reduce": "mean",
            "empty_val": "nan",
            "oname": "f1_mean"
        },
        {
            "fnames": ["f1", "f2", "f3"],
            "reduce": "mean",
            "empty_val": "nan",
            "oname": "f1_f2_f3_mean"
        },
        {
            "fnames": ["f4"],
            "target_val": 1,
            "reduce": "binary_mask",
            "count_threshold": 2,
            "empty_val": "nan",
            "oname": "bin_mask"
        }
    ]
}

# ---  UTIL FUNCTIONS  --- #
# ------------------------ #
def _load_asset(asset):
    return np.loadtxt("tests/assets/" + asset, delimiter=",")


# ---  BEFORE TEST LOGIC  --- #
# --------------------------- #
@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    main_logger_init()


# ---   TEST CLASS   --- #
# ---------------------- #
class TestRasterGridEval:
    """
    :author: Alberto M. Esmoris Pena

    Unit test class for :class:`.RasterGridEvaluator`.
    """
    # ---   TEST METHODS   --- #
    # ------------------------ #
    def test_sequential(self):
        # Load data
        pcloud_in = self.point_cloud_from_file("3DPCloud_for_raster_eval.xyz")
        rasters_ref = self.raster_from_file("2DRaster_for_raster_eval")
        # Compute raster grid evaluation
        evaluator = RasterGridEvaluator(**SPEC)
        evaluation = evaluator.eval(pcloud_in)
        rasters_out = evaluation.Fgrids
        # Assert generated raster grid
        self.assert_rasters(rasters_out, rasters_ref)

    def test_parallel(self):
        pass  # TODO Rethink : Implement

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    def point_cloud_from_file(self, path):
        X = _load_asset(path)
        X, F = X[:, :3], X[:, 3:]
        return PointCloudFactoryFacade.make_from_arrays(X, F)

    def raster_from_file(self, prepath):
        Fgrids = [
            _load_asset(prepath+f'_{case}.csv')
            for case in ['A1', 'B1', 'B2', 'B3', 'C1']
        ]
        return [
            np.array([Fgrids[0]]),
            np.array([Fgrids[1], Fgrids[2], Fgrids[3]]),
            np.array([Fgrids[-1]])
        ]

    def assert_rasters(self, raster_out, raster_ref):
        for out_grid, ref_grid in zip(raster_out, raster_ref):
            assert out_grid.shape == ref_grid.shape
            assert np.allclose(out_grid, ref_grid, equal_nan=True)
