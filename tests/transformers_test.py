import pytest
import numpy as np
from src.utils.ftransf.pca_transformer import PCATransformer
from src.main.main_logger import main_logger_init
from src.utils.ftransf.standardizer import Standardizer

features = [
    [
        -1.5198752,
        3.0611641,
        0.83759266,
        0.7155187,
        0.13851555,
        1.1356975,
        -0.4616439,
        0.40397194,
        -0.01597583,
        -0.62195075,
        0.52257407,
        0.3289029,
    ],
    [
        -1.5458847,
        3.1264014,
        0.72809094,
        0.74452454,
        0.43026692,
        1.0937977,
        -0.5291867,
        0.50502616,
        -0.05155919,
        -0.6755433,
        0.23294379,
        0.48922414,
    ],
    [
        -1.260156,
        3.305099,
        0.5581642,
        1.1453952,
        0.72799885,
        1.350359,
        -0.86501706,
        0.6850138,
        0.09600829,
        -0.642268,
        -0.27398744,
        0.50672716,
    ],
    [
        -2.1285276,
        6.519185,
        1.4655763,
        1.973307,
        0.44348264,
        0.08070929,
        0.46813357,
        -0.51722664,
        -0.94485784,
        1.1758758,
        -0.09054696,
        -0.6194549,
    ],
    [
        -1.1037253,
        5.751667,
        1.3254039,
        1.1918845,
        1.1079161,
        0.75074494,
        -1.1429025,
        0.50691587,
        -1.0069766,
        0.8298567,
        -0.04356513,
        0.24081056,
    ],
    [
        0.9697391,
        2.7870307,
        0.66030866,
        1.4710906,
        0.43698433,
        -1.028134,
        -1.0244044,
        -0.5329787,
        1.3878137,
        -0.2318638,
        0.17802665,
        0.41517264,
    ],
]


def _load_asset(asset):
    return np.loadtxt("tests/assets/" + asset, delimiter=",")


@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    main_logger_init()


class TestTransformers:
    """
    Unit test class for the :class:`.PCATransformer`.
    """

    def test_standardizer(self):
        transformer = Standardizer(fnames=["AUTO"])
        out = transformer.transform(
            F=np.array(features), y=[], fnames=["AUTO"], out_prefix=None
        )

        # np.savetxt("Standardizer_test.csv", out, delimiter=",")

        expected_out = _load_asset("Standardizer_test.csv")

        assert np.allclose(np.array(out), expected_out)

    def test_pca_transformer(self):
        """
        Test case for the PCA transformer.

        Returns:
            None
        """
        # Add your assertions here
        transformer = PCATransformer(out_dim=0.99, whiten=False)

        out = transformer.transform(
            F=np.array(features), y=[""], fnames=["AUTO"], out_prefix=None
        )

        # np.savetxt("pca_transformer_test.csv", out, delimiter=",")

        expected_out = _load_asset("PCATransformer_test.csv")

        assert np.allclose(out, expected_out)

    def test_standardizer_invalid_input(self):
        transformer = Standardizer(fnames=["AUTO"])
        with pytest.raises((AttributeError, ValueError)):
            transformer.transform(F=None, y=[], fnames=["AUTO"], out_prefix=None)

    def test_pca_transformer_invalid_input(self):
        transformer = PCATransformer(out_dim=0.99, whiten=False)
        with pytest.raises((AttributeError, ValueError)):
            transformer.transform(F=None, y=[""], fnames=["AUTO"], out_prefix=None)
