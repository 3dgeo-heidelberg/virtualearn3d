# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
from src.utils.str_utils import StrUtils
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
import src.main.main_logger as LOGGING
from scipy.spatial.kdtree import KDTree as KDT
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class FPSDecoratorTransformer:
    r"""
    :author: Alberto M. Esmoris Pena

    Class representing an FPS transformer that can be used to decorate
    different components of the framework.

    A point cloud :math:`\pmb{P} = [\pmb{X} | \pmb{F} | \pmb{y}]` can be
    reduced to an FPS representation
    :math:`\pmb{P'} = [\pmb{X'} | \pmb{F'} | \pmb{y'}]`. More concretely,
    an input point cloud with dimensionalities
    :math:`\pmb{X} \in \mathbb{R}^{m \times n_x}`,
    :math:`\pmb{F} \in \mathbb{R}^{m \times n_f}`,
    :math:`\pmb{y} \in \mathbb{Z}^{m}`
    will be transformed to a representation with dimensionalities
    :math:`\pmb{X'} \in \mathbb{R}^{R \times n_x}`,
    :math:`\pmb{F'} \in \mathbb{R}^{R \times n_f}`,
    :math:`\pmb{y'} \in \mathbb{Z}^{R}`
    where :math:`R \leq m`.

    :ivar num_points: The number of points :math:`R` the input points must be
        reduced to.
        In other words, for a given number of input points :math:`m_1`,
        the reduced number of points will be :math:`R`. For another,
        let us say different (i.e., :math:`m_1 \neq m_2`) number of
        points, the reduced number of points will also be :math:`R`.
        Alternatively, it can be given as a string. If so, it is
        understood as an expression that can be evaluated, where "m"
        is the number of input points. For instance: "m/2" means the
        FPS will consider half of the input points.
    :vartype num_points: int or str
    :ivar num_encoding_neighbors: How many closest neighbors consider when
        doing reductions.
        For instance, for three encoding neighbors reducing a value
        means three points in the original domain will be considered
        to estimate the value in the representation domain.
        A value of zero means the representation will be generated
        but no neighborhood information connecting the points from
        the original domain to the representation domain will be
        computed.
    :vartype num_encoding_neighbors: int
    :ivar num_decoding_neighbors: How many closest neighbors consider when
        doing propagations.
        For instance, for three decoding neighbors propagating a value
        means three points in the representation domain will be
        considered to estimate the value in the original domain.
        A value of zero means the representation will be generated
        but no neighborhood information connecting the points from
        the representation domain back to the original domain will
        be computed.
    :vartype num_decoding_neighbors: int
    :ivar release_encoding_neighborhoods: A flag to enable releasing the
        encoding neighborhoods after building the representation.
        It can be used to save memory when the topological information of the
        encoding process will not be used after generating the representation.
    :vartype release_encoding_neighbors: int
    :ivar fast: A flag to enable the fast-computation mode. When True, a random
        uniform subsampling will be computed before the furthest point
        sampling so the latest is faster because it is not considering
        the entire input point cloud.
    :vartype fast: bool
    :ivar threads: The number of threads to be used for parallel computations.
        A value of -1 means using all available cores.
    :vartype threads: int
    :ivar representation_report_path: Path where a point cloud with the points
        in the representation space will be exported.
    :vartype representation_report_path: str
    :ivar N: The encoding neighborhoods that define what points in the
        original space must be considered to obtain each point in the
        representation space. It is a matrix
        :math:`\pmb{N} \in \mathbb{R}^{R \times k_e}` for :math:`k_e` encoding
        neighbors where each row corresponds to a point in the reduced
        space and the columns give the indices of their neighbors in the
        original space.
    :vartype N: :class:`np.ndarray`
    :ivar M: The decoding neighborhoods that define what points in the
        representation space must be considered to propagate back to the
        original space. It is a matrix
        :math:`\pmb{M} \in \mathbb{R}^{m \times k_d}` for :math:`k_d` decoding
        neighbors where each row corresponds to a point in the original
        space and the columns give the indices of their neighbors in the
        reduced space.
    :vartype M: :class:`np.ndarray`
    """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        r"""
        Initialize/instantiate a receptive field object.

        :param kwargs: The key-word specification to instantiate the
            FPSDecoratorTransformer.

        :Keyword Arguments:
            *   *num_points* (``int`` or ``str``) --
                The number of points :math:`R` the input points must be reduced
                to.
                In other words, for a given number of input points :math:`m_1`,
                the reduced number of points will be :math:`R`. For another,
                let us say different (i.e., :math:`m_1 \neq m_2`) number of
                points, the reduced number of points will also be :math:`R`.
                Alternatively, it can be given as a string. If so, it is
                understood as an expression that can be evaluated, where "m"
                is the number of input points. For instance: "m/2" means the
                FPS will consider half of the input points.
            * *num_encoding_neighbors* (``int``) --
                How many closest neighbors consider when doing reductions.
                For instance, for three encoding neighbors reducing a value
                means three points in the original domain will be considered
                to estimate the value in the representation domain.
                A value of zero means the representation will be generated
                but no neighborhood information connecting the points from
                the original domain to the representation domain will be
                computed.
            * *num_decoding_neighbors* (``int``) --
                How many closest neighbors consider when doing propagations.
                For instance, for three decoding neighbors propagating a value
                means three points in the representation domain will be
                considered to estimate the value in the original domain.
                A value of zero means the representation will be generated
                but no neighborhood information connecting the points from
                the representation domain back to the original domain will
                be computed.
            * *release_encoding_neighborhoods* (``bool``) --
                A flag to enable releasing the encoding neighborhoods after
                building the representation. It can be used to save memory when
                the topological information of the encoding process will not
                be used after generating the representation.
            * *fast* (``bool``) --
                A flag to enable the fast-computation mode. When True, a random
                uniform subsampling will be computed before the furthest point
                sampling so the latest is faster because it is not considering
                the entire input point cloud.
            * *threads* (``int``) --
                The number of threads to be used for parallel computations.
                Giving -1 means using all available cores.
            * *representation_report_path* (``str``) --
                Path where a point cloud with the points in the representation
                space will be exported.
        """
        # Assign attributes
        self.num_points = kwargs.get('num_points', None)
        self.num_encoding_neighbors = kwargs.get('num_encoding_neighbors', 0)
        self.num_decoding_neighbors = kwargs.get('num_decoding_neighbors', 0)
        self.release_encoding_neighborhoods = kwargs.get(
            'release_encoding_neighborhoods', False
        )
        self.fast = kwargs.get('fast', False)
        self.threads = kwargs.get('threads', -1)
        self.representation_report_path = kwargs.get(
            'representation_report_path', None
        )
        # Initialize attributes that will be generated when transforming
        self.N = None  # Neighborhood matrix with indices in the original space
        self.M = None  # Neighborhood matrix with indices in the representation
        # Validate number of points
        if self.num_points is None or (
            isinstance(self.num_points, int) and self.num_points < 1
        ) or not isinstance(self.num_points, (str, int)):
            raise VL3DException(
                f'FPSDecoratorTransformer cannot work for {self.num_points} '
                'target number of points.'
            )
        # Validate number of encoding neighbors
        if self.num_encoding_neighbors is None or self.num_encoding_neighbors < 0:
            raise VL3DException(
                'FPSDecoratorTransformer cannot work for '
                f'{self.num_encoding_neighbors} encoding neighbors.'
            )
        # Validate number of decoding neighbors
        if self.num_decoding_neighbors is None or self.num_decoding_neighbors < 0:
            raise VL3DException(
                'FPSDecoratorTransformer cannot work for '
                f'{self.num_decoding_neighbors} decoding neighbors.'
            )

    # ---   FPS DECORATOR METHODS   --- #
    # --------------------------------- #
    def transform(self, X, F=None, y=None, out_prefix=None):
        """
        Transform the given point cloud to its FPS representation.

        :param X: The structure space matrix (i.e., the matrix of coordinates).
        :param F: The feature space matrix (i.e., the matrix of features).
        :param y: The reference classes (i.e., the classification).
        :param out_prefix: The output prefix (OPTIONAL). It might be used by a
            report to particularize the output path.
        :return: The structure space matrix, the feature space matrix, and the
            classes vector of the FPS representation.
        :rtype: tuple (:class:`np.ndarray`,
                :class:`np.ndarray` or None,
                :class:`np.ndarray` or None)
        """
        # Handle num points
        num_points = self.eval_num_points(X=X)
        # Build support points
        if self.fast:
            rep_X = np.array(X)
            np.random.shuffle(rep_X)
            rep_X = rep_X[:num_points]
        else:
            rep_X = ReceptiveFieldFPS.compute_fps_on_3D_pcloud(
                X,
                num_points=num_points,
                fast=False
            )
        # Compute encoding neighborhoods
        if self.num_encoding_neighbors > 0:
            kdt = KDT(X)
            self.N = kdt.query(
                rep_X,
                k=self.num_encoding_neighbors,
                workers=self.threads
            )[1]
        # Compute decoding neighborhoods
        if self.num_decoding_neighbors > 0:
            kdt = KDT(rep_X)
            self.M = kdt.query(
                X,
                k=self.num_decoding_neighbors,
                workers=self.threads
            )[1]
        # Encode features for representation
        rep_F = None if F is None else self.reduce(F)
        # Encode classes for representation
        rep_y = None if y is None else self.reduce(y)
        # Release encoding neighborhoods if requested
        if self.release_encoding_neighborhoods:
            self.N = None
        # Export representation, if requested
        if self.representation_report_path is not None:
            representation_report_path = self.representation_report_path \
                if out_prefix is None else \
                out_prefix[:-1] + self.representation_report_path[1:]
            rep_pcloud = PointCloudFactoryFacade.make_from_arrays(
                rep_X, rep_F, y=rep_y
            )
            PointCloudIO.write(rep_pcloud, representation_report_path)
            LOGGING.LOGGER.info(
                'FPS representation exported to '
                f'"{representation_report_path}".'
            )
        # Return
        return rep_X, rep_F, rep_y

    def transform_pcloud(
        self, pcloud, fnames=None, ignore_y=False, out_prefix=None
    ):
        """
        Transform the given point cloud to its FPS representation.

        :param pcloud: The point cloud to be transformed.
        :type pcloud: :class:`.PointCloud`
        :param fnames: A list with the names of the features to be considered.
            If an empty list is given, no features will be considered.
            If None is given, all features will be considered.
        :type fnames: None or list of str
        :param ignore_y: Whether to ignore the classes (True) or to encode
            them in the representation (False).
        :type ignore_y: bool
        :param out_prefix: The output prefix (OPTIONAL). It might be used by a
            report to particularize the output path.
        :type out_prefix: None or str
        :return: A new point cloud that is an FPS representation of the input
            one.
        :rtype: :class:`.PointCloud`
        """
        # Get coordinates
        X = pcloud.get_coordinates_matrix()
        # Handle features
        if fnames is not None and len(fnames) == 0:
            F = None
        else:
            if fnames is None:
                F = pcloud.get_features_matrix(pcloud.get_features_names())
            else:
                F = pcloud.get_features_matrix(fnames)
        # Handle classes
        y = None
        if not ignore_y:
            y = pcloud.get_classes_vector() if pcloud.has_classes() else None
        # Transform and return
        rep_X, rep_F, rep_y = self.transform(
            X=X, F=F, y=y, out_prefix=out_prefix
        )
        return PointCloudFactoryFacade.make_from_arrays(
            rep_X, rep_F, y=rep_y, header=pcloud.get_header(),
            fnames=pcloud.get_features_names() if fnames is None else fnames
        )

    def propagate(self, rep_x):
        r"""
        Propagate a representation on :math:`R` back to the original space
        of :math:`m` points.

        :param rep_x: The representation to be propagated back to the original
            space. When given as a matrix, rows must be points and columns
            must be point-wise values.
        :type rep_x: :class:`np.ndarray`
        :return: The propagated representation.
        :rtype: :class:`np.ndarray`
        """
        # Nearest neighbor reduction
        if self.num_decoding_neighbors == 1:
            return rep_x[self.M]
        # Mean of the neighborhood reduction for 1 feature
        if len(rep_x.shape) == 1:
            return np.mean(rep_x[self.M], axis=1)
        # Mean of the neighborhood reduction for >1 features
        return np.mean(rep_x[self.M].T, axis=1).T

    def reduce(self, x):
        r"""
        Reduce the original space of :math:`m` points to a representation
        space of :math:`R` points.

        :param x: The original values to be reduced to the representation
            space. When given as a matrix, rows must be points and columns
            must be point-wise values.
        :type x: :class:`np.ndarray`
        :return: The reduced values.
        :rtype: :class:`np.ndarray`
        """
        # Nearest neighbor reduction
        if self.num_encoding_neighbors == 1:
            return x[self.N]
        # Mean of the neighborhood reduction for 1 feature
        if len(x.shape) == 1:
            return np.mean(x[self.N], axis=1)
        # Mean of the neighborhood reduction for >1 features
        return np.mean(x[self.N].T, axis=1).T

    def eval_num_points(self, X=None, m=None):
        """
        Evaluate the number of points if it is a string and return the result,
        otherwise return the number directly.

        :param X: The structure space matrix representing the point cloud (rows
            are columns),
        :type X: :class:`np.ndarray` or None
        :param m: The number of points (when not given, the number of rows in
            X will be considered to initialize m) in the point cloud.
        :type m: int or None
        :return: The number of points that must be obtained after applying
            the transformation.
        :rtype: int
        """
        num_points = self.num_points
        if isinstance(num_points, str):  # If string, evaluate expression
            if m is None:
                m = X.shape[0]  # Number of input points (for the expression)
            num_points = max(1, int(eval(StrUtils.to_numpy_expr(num_points))))
        return num_points
