# ---   IMPORTS   --- #
# ------------------- #
from src.report.report import Report, ReportException
from src.inout.io_utils import IOUtils
from src.inout.point_cloud_io import PointCloudIO
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
import numpy as np
import os


# ---   CLASS   --- #
# ----------------- #
class FeaturesStructuringLayerReport(Report):
    """
    :author: Alberto M. Esmoris Pena

    Class to handle reports that represent a features structuring layer.

    See :class:`.Report`.
    See also :class:`.FeaturesStructuringLayer`.

    :ivar QX: The structure space matrix of the features structuring kernel.
    :vartype QX: :class:`np.ndarray`
    :ivar omegaF: The vector feature-wise weights.
    :vartype omegaF: :class:`np.ndarray`
    :ivar omegaD: The vector of distance-wise weights.
    :vartype omegaD: :class:`np.ndarray`
    :ivar omegaD_name: The name of the omegaD vector. It can be overriden to
        utilize the report for a :class:`.RBFFeatExtractLayer`.
    :vartype omegaD_name: str
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, QX, omegaF, omegaD, **kwargs):
        """
        Initialize an instance of FeaturesStructuringLayerReport.

        :param kwargs: The key-word arguments.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.QX = QX
        self.omegaF = omegaF
        self.omegaD = omegaD
        self.QXpast = kwargs.get('QXpast', None)
        # Might be overriden to reuse the report
        self.QX_name = kwargs.get('QX_name', 'QX')
        self.omegaD_name = kwargs.get('omegaD_name', '$\\omega_{Di}')
        # Validate
        if self.QX is None and self.omegaF is None and self.omegaD is None:
            raise ReportException(
                'FeaturesStructuringLayerReport did not receive input data '
                'at all.'
            )

    # ---   TO FILE   --- #
    # ------------------- #
    def to_file(self, path, out_prefix=None):
        """
        Write the report (point cloud, and ASCII vectors) to files.

        :param path: Path to the directory where the report files must be
            written.
        :type path: str
        :param out_prefix: The output prefix to expand the path (OPTIONAL).
        :type out_prefix: str
        :return: Nothing, the output is written to a file.
        """
        # Expand path if necessary
        if out_prefix is not None and path[0] == "*":
            path = out_prefix[:-1] + path[1:]
        # Check
        IOUtils.validate_path_to_directory(
            os.path.dirname(path),
            'Cannot find the directory to write the features structuring '
            'layer representation'
        )
        # Compute QX distances vector between past and now
        if self.QX is not None:
            QXdiff = None
            if self.QXpast is not None:
                QXdiff = np.sqrt(
                    np.sum(np.power(self.QX-self.QXpast, 2), axis=1)
                )
            # Output QX point cloud
            fnames = ['distance_weight']
            outF = self.omegaD.reshape((-1, 1))
            if QXdiff is not None:
                fnames.append('l2_diff')
                outF = np.hstack([outF, QXdiff.reshape((-1, 1))])
            PointCloudIO.write(
                PointCloudFactoryFacade.make_from_arrays(
                    self.QX,
                    outF,
                    None,
                    fnames=fnames
                ),
                os.path.join(path, f'{self.QX_name}.laz')
            )
        # Output omegaD vector
        if self.omegaD is not None:
            np.savetxt(
                os.path.join(path, f'{self.omegaD_name}.csv'),
                self.omegaD,
                fmt='%.7f'
            )
        # Output omegaF vector
        if self.omegaF is not None:
            np.savetxt(
                os.path.join(path, 'omegaF.csv'),
                self.omegaF,
                fmt='%.7f'
            )
