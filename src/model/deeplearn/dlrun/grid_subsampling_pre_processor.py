# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ptransf.receptive_field_gs import ReceptiveFieldGS
from src.utils.ptransf.receptive_field_fps import ReceptiveFieldFPS
from src.model.deeplearn.dlrun.receptive_field_pre_processor import \
    ReceptiveFieldPreProcessor
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.pcloud.point_cloud_factory_facade import PointCloudFactoryFacade
from src.inout.point_cloud_io import PointCloudIO
import src.main.main_logger as LOGGING
from scipy.spatial import KDTree as KDT
import scipy.stats
import numpy as np
import joblib
import time


# ---   CLASS   --- #
# ----------------- #
class GridSubsamplingPreProcessor(ReceptiveFieldPreProcessor):
    r"""
    :author: Alberto M. Esmoris Pena

    Preprocess the input dictionary of X (coordinates), F (features), and y
    (expected values) so it can be feed to some neural networks such as
    PointNet.

    See :class:`.ReceptiveFieldGS`.
    See :class:`.ReceptiveFieldPreProcessor`.

    :ivar sphere_radius: The radius of the sphere that bounds a neighborhood.
        For an arbitrary set of 3D points it can be found as follows:

        .. math::
            r = \dfrac{1}{2} \, \max \; \biggl\{
                x_{\mathrm{max}}-x_{\mathrm{min}},
                y_{\mathrm{max}}-y_{\mathrm{min}},
                z_{\mathrm{max}}-z_{\mathrm{min}}
            \biggr\}

    :vartype sphere_radius: float
    :ivar separation_factor: How many times the sphere radius separates the
        support points to find the input neighborhoods that will be used to
        generate the receptive fields for the neural network.

        For a given separation factor :math:`k`, the following condition
        should be satisfied to prevent missing any region of the input
        point cloud on a :math:`n`-dimensional space:

        .. math::
            k \leq \dfrac{2}{\sqrt{n}}

    :vartype separation_factor: float
    :ivar cell_size: The cell size defining the receptive field. See
        :class:`.ReceptiveFieldGS`.
    :vartype cell_size: :class:`np.ndarray`
    :ivar interpolate: Flag to control whether to interpolate the missing
        centroids of the receptive fields (True) or not (False). When
        interpolating, centroids will be computed from missing centroids.
        Otherwise, all missing centroids will be replaced by the point
        obtained as the coordinate-wise mean.
    :vartype interpolate: bool
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialization/instantiation of a Grid Subsampling pre-processor.

        :param kwargs: The key-word arguments for the
            GridSubSamplingPreProcessor.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.sphere_radius = kwargs.get('sphere_radius', 1.0)
        self.separation_factor = kwargs.get('separation_factor', np.sqrt(3)/4)
        self.cell_size = np.array(kwargs.get('cell_size', [0.1, 0.1, 0.1]))
        self.interpolate = kwargs.get('interpolate', True)

    # ---   RUN/CALL   --- #
    # -------------------- #
    def __call__(self, inputs):
        r"""
        Executes the pre-processing logic. It also updates the cache-like
        variables of the preprocessor.

        The pre-processing logic consists of steps:

        **1) Generate support points.**

        Support points are generated as a set of points separated in :math:`kr`
        units between them and covering the entire point cloud. Watch out,
        generating support points with
        :math:`k > 2/\sqrt{n}` in a :math:`n`-dimensional
        space leads to gaps between the bounding spheres centered on the
        support points.

        **2) Find neighborhoods centered on support points.**
        For each support point, a neighborhood (by default a spherical
        neighborhood but the neighborhood definition can be arbitrarily
        changed) centered on it is found.

        **3) Filter empty neighborhoods.**
        Any support point that leads to an empty neighborhood when considering
        as neighbors points from the input point cloud is filtered out.

        **4) Transform non-empty neighborhoods to receptive fields.**
        Each non-empty neighborhood is transformed to a receptive field.
        See :class:`.ReceptiveFieldGS` for further details.

        :param inputs: A key-word input where the key "X" gives the input
            dataset and the "y" (OPTIONALLY) gives the reference values that
            can be used to fit/train a PointNet model.
        :type inputs: dict
        :return: Either (Xout, yout) or Xout. Where Xout are the points
            representing the receptive field and yout (only given when
            "y" was given in the inputs dictionary) the corresponding
            reference values for those points.
        """
        # Extract inputs
        start = time.perf_counter()
        X, y = inputs['X'], inputs.get('y', None)
        # Extract neighborhoods
        sup_X, I = self.find_neighborhood(X, y=y)
        # Remove empty neighborhoods and corresponding support points
        I, sup_X = GridSubsamplingPreProcessor.clean_support_neighborhoods(
            sup_X, I
        )
        # Export support points if requested
        if(
            inputs.get('training_support_points', False) and
            self.training_support_points_report_path is not None
        ):
            GridSubsamplingPreProcessor.support_points_to_file(
                sup_X,
                self.training_support_points_report_path
            )
        if(
            inputs.get('support_points', False) and
            self.support_points_report_path
        ):
            GridSubsamplingPreProcessor.support_points_to_file(
                sup_X,
                self.support_points_report_path
            )
        self.last_call_neighborhoods = I
        # Prepare receptive field
        self.last_call_receptive_fields = [
            ReceptiveFieldGS(
                bounding_radii=np.array([
                    self.sphere_radius for j in range(X.shape[1])
                ]),
                cell_size=self.cell_size
            )
            for Ii in I
        ]
        self.last_call_receptive_fields = joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(
                self.last_call_receptive_fields[i].fit
            )(
                X[Ii], sup_X[i]
            )
            for i, Ii in enumerate(I)
        )
        # Neighborhoods ready to be fed into the neural network
        Xout = np.array(joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(
                self.last_call_receptive_fields[i].centroids_from_points
            )(
                X[Ii],
                interpolate=self.interpolate,
                fill_centroid=not self.interpolate
            )
            for i, Ii in enumerate(I)
        ))
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'The grid subsampling pre processor generated {Xout.shape[0]} '
            'receptive fields. '
        )
        if y is not None:
            yout = self.reduce_labels(Xout, y, I=I)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                f'The grid subsampling pre processor pre-processed '
                f'{X.shape[0]} points for training in {end-start:.3f} seconds.'
            )
            return Xout, yout
        LOGGING.LOGGER.info(
            f'The grid subsampling pre processor pre-processed {X.shape[0]} '
            f'points for predictions in {end-start:.3f} seconds.'
        )
        return Xout

    # ---   POINT-NET METHODS   --- #
    # ----------------------------- #
    def get_num_input_points(self):
        """
        See
        :meth:`point_net_pre_processor.PointNetPreProcessor.get_num_input_points`
        .
        """
        return ReceptiveFieldGS.num_cells_from_cell_size(self.cell_size)

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def build_support_points(
        X, separation_factor, sphere_radius,
        y=None, class_distr=None, center_on_X=False,
        support_strategy='grid', support_strategy_num_points=1000,
        support_strategy_fast=False,
        nthreads=1
    ):
        r"""
        Compute the support points separated :math:`k` times the radius
        :math:`r` distributed along the bounding box defining the boundaries
        of :math:`\pmb{X}`.

        Alternatively, if the labels :class:`\pmb{y}` are passed and a given
        class distribution is requested, the support points are selected to
        match this distribution.

        :param X: `\pmb{X}`, i.e., the matrix of coordinates representing the
            input points.
        :param separation_factor: :math:`k`
        :param sphere_radius: :math:`r`
        :param y: The vector of point-wise labels (OPTIONAL).
        :param class_distr: The vector of class-wise distribution (OPTIONAL).
        :param center_on_X: When True, the support points will be points taken
            from X (as the nearest neighbors of the initial support points).
            Otherwise, they will be automatically computed such that they
            do not necessarily correspond to points in X.
        :param support_strategy: By default, "grid", which means the support
            points will be taken by grid sampling. It can be "fps" to apply
            furthest point sampling. The support strategy will be ignored if
            class_distr is given.
        :param support_strategy_num_points: The number of points to be
            considered when using a furthest point sampling support strategy.
        :param nthreads: How many threads use for parallel computations, if
            any.
        :return: The support points as a matrix where rows are support points
            and columns are coordinates.
        :rtype: :class:`np.ndarray`
        """
        # Validate y and class_distr are consistent
        if y is None and class_distr is not None:
            raise DeepLearningException(
                'Support points cannot be built from a given class '
                'distribution when class labels are not available.'
            )
        # Build support points considering point-wise classes
        if y is not None and class_distr is not None:
            LOGGING.LOGGER.debug(
                'Support points are built from a given class distribution.'
            )
            idx_by_class = [  # Obtain indices by classes
                np.flatnonzero(y == cidx) for cidx in range(len(class_distr))
            ]
            for i in range(len(idx_by_class)):  # Shuffle and truncate
                np.random.shuffle(idx_by_class[i])  # Random shuffle
                idx_by_class[i] = idx_by_class[i][:class_distr[i]]  # Truncate
            # Extract and shuffle support points
            sup_X = np.vstack([
                X[idx_by_class[i]] for i in range(len(idx_by_class))
            ])
            np.random.shuffle(sup_X)
            # Return support points from point-wise classes
            return sup_X
        # Build support points without considering point-wise classes
        LOGGING.LOGGER.debug(
            'Support points are built without considering point-wise classes.'
        )
        # Grid of support points strategy
        support_strategy_low = support_strategy.lower()
        if support_strategy_low == 'grid':
            xmin, xmax = np.min(X, axis=0), np.max(X, axis=0)
            l = separation_factor * sphere_radius  # Cell size
            sup_X = np.meshgrid(
                *[
                    np.concatenate([
                        np.arange(xmin[j], xmax[j], l), [xmax[j]]
                    ])
                    for j in range(X.shape[1])
                ]
            )
            sup_X = np.array([Gi.flatten() for Gi in sup_X]).T
        # Support points by furthest point sampling
        elif support_strategy_low == 'fps':
            # Compute the FPS
            sup_X = ReceptiveFieldFPS.compute_fps_on_3D_pcloud(
                X,
                num_points=support_strategy_num_points,
                fast=support_strategy_fast
            )
            center_on_X = False  # Not necessary when using FPS
        else:
            raise DeepLearningException(
                'Support points cannot be built with support strategy '
                f'"{support_strategy}".'
            )
        # Post-process
        if center_on_X:  # Support points must correspond to points in X
            LOGGING.LOGGER.debug(
                'Support points are centered on the point cloud.'
            )
            kdt = KDT(X)
            D, I = kdt.query(
                sup_X,
                k=1,
                distance_upper_bound=sphere_radius,  # Prune when dnn>r
                workers=nthreads
            )
            mask = I != len(X)  # Filter points further than r wrt nn
            I = np.unique(I[mask])
            return X[I]
        # Return original support points (dont need to match points in X)
        return sup_X

    @staticmethod
    def clean_support_neighborhoods(sup_X, I):
        """
        Compute the clean version of the given support neighborhoods, i.e.,
        support points in sup_X and their neighborhoods as defined in I but
        considering only non-empty neighborhoods.

        :param sup_X: The matrix of coordinates representing the support
            points.
        :param I: The indices (in the original point domain) corresponding
            to each support point. In other words, I[i] gives the indices
            in X of the support point i in X_sup.
        :return: The clean matrix of coordinates representing the support
            points and their neighborhoods.
        :rtype: tuple
        """
        non_empty_mask = [len(Ii) > 0 for Ii in I]
        I = [Ii for i, Ii in enumerate(I) if non_empty_mask[i]]
        sup_X = sup_X[non_empty_mask]
        return I, sup_X

    @staticmethod
    def support_points_to_file(sup_X, path):
        """
        Export the given support points to a LAS/LAZ point cloud at the given
        path.

        :param sup_X: The support points to be exported.
        :param path: The path where the LAS/LAZ file must be written.
        :return: Nothing at all, but an output file is generated.
        """
        sup_pcloud = PointCloudFactoryFacade.make_from_arrays(sup_X, None)
        PointCloudIO.write(sup_pcloud, path)
        LOGGING.LOGGER.info(f'Support points exported to "{path}".')

    def reduce_labels(self, X_rf, y, I=None):
        r"""
        Reduce the given labels :math:`\pmb{y} \in \mathbb{Z}_{\geq 0}^{m}`
        to the receptive field labels
        :math:`\pmb{y}_{\mathrm{rf}} \in \mathbb{Z}_{\geq 0}^{R}`.

        :param X_rf: The matrices of coordinates representing the receptive
            fields.
        :type X_rf: :class:`np.ndarray`
        :param y: The labels of the original point cloud that must be reduced
            to the receptive field.
        :type y: :class:`np.ndarray`
        :param I: The list of neighborhoods. Each element of I is itself a list
            of indices that represents the neighborhood in the point cloud
            that corresponds to the point in the receptive field.
        :type I: list
        :return: The reduced labels for each receptive field.
        """

        # Handle automatic neighborhoods from cache
        if I is None:
            I = self.last_call_neighborhoods
        # Validate neighborhoods are given
        if I is None or len(I) < 1:
            raise DeepLearningException(
                'GridSubsamplingPreProcessor cannot reduce labels because '
                'no neighborhood indices were given.'
            )
        # Compute and return the reduced labels
        return np.array(joblib.Parallel(n_jobs=self.nthreads)(
            joblib.delayed(
                self.last_call_receptive_fields[i].reduce_values
            )(
                X_rf[i],
                y[Ii],
                reduce_f=lambda x: scipy.stats.mode(x)[0][0],
                fill_nan=True
            ) for i, Ii in enumerate(I)
        ))

    def find_neighborhood(self, X, y=None):
        r"""
        Find the requested neighborhoods in the given input point cloud
        represented by the matrix of coordinates :math:`\pmb{X}`.

        :param X: The matrix of coordinates.
        :type X: :class:`np.ndarray`
        :param y: The vector of expected values (generally, class labels).
            It is an OPTIONAL argument that is only necessary when the
            neighborhoods must be found following a given class distribution.
        :type y: :class:`np.ndarray`
        :return: A tuple which first element are the support points
            representing the centers of the neighborhoods and which second
            element is a list of neighborhoods, where each neighborhood is
            represented by a list of indices corresponding to the rows (points)
            in :math:`\pmb{X}` that compose the neighborhood.
        :rtype: tuple
        """
        # Handle neighborhood finding
        class_distr = self.training_class_distribution if y is not None \
            else None
        sup_X = GridSubsamplingPreProcessor.build_support_points(
            X,
            self.separation_factor,
            self.sphere_radius,
            y=y,
            class_distr=class_distr,
            center_on_X=self.center_on_pcloud,
            nthreads=self.nthreads
        )
        kdt = KDT(X)
        kdt_sup = KDT(sup_X)
        I = kdt_sup.query_ball_tree(kdt, self.sphere_radius)  # Neigh. indices
        # Return found neighborhood
        return sup_X, I

    # ---   OTHER METHODS   --- #
    # ------------------------- #
    def overwrite_pretrained_model(self, spec):
        """
        See
        :meth:`point_net_pre_processor.PointNetPreProcessor.overwrite_pretrained_model`
        method.
        """
        # Overwrite from parent
        super().overwrite_pretrained_model(spec)
        spec_keys = spec.keys()
        # Overwrite the attributes of the grid subsampling pre-processor
        if 'sphere_radius' in spec_keys:
            self.sphere_radius = spec['sphere_radius']
        if 'separation_factor' in spec_keys:
            self.separation_factor = spec['separation_factor']
        if 'cell_size' in spec_keys:
            self.cell_size = spec['cell_size']
        if 'interpolate' in spec_keys:
            self.interpolate = spec['interpolate']

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized grid subsampling
        pre-processor.

        See :meth:`ReceptiveFieldPreProcessor.__getstate__`.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Obtain parent's state
        state = super().__getstate__()
        # Update state
        state['sphere_radius'] = self.sphere_radius
        state['separation_factor'] = self.separation_factor
        state['cell_size'] = self.cell_size
        state['interpolate'] = self.interpolate
        # Return state
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized grid subsampling pre-processor.

        See :meth:`ReceptiveFieldPreProcessor.__setstate__`.

        :param state: The state's dictionary of the saved grid subsampling
            pre-procesor.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Call parent
        super().__setstate__(state)
        # Assign member attributes from state
        self.sphere_radius = state['sphere_radius']
        self.separation_factor = state['separation_factor']
        self.cell_size = state['cell_size']
        self.interpolate = state['interpolate']
