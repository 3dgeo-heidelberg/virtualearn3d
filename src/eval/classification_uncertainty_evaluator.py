# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluator import Evaluator, EvaluatorException
from src.eval.classification_uncertainty_evaluation import \
    ClassificationUncertaintyEvaluation
from src.model.classification_model import ClassificationModel
from src.model.deeplearn.point_net_pwise_classif_model import \
    PointNetPwiseClassifModel
from src.model.deeplearn.rbf_net_pwise_classif_model import \
    RBFNetPwiseClassifModel
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class ClassificationUncertaintyEvaluator(Evaluator):
    r"""
    :author: Alberto M. Esmoris Pena

    Class to evaluate classification-like predictions to analyzer their
    uncertainty.

    :ivar class_names: The name for each class.
    :vartype class_names: list
    :ivar include_probabilities: Whether to include the probabilities in the
        resulting evaluation (True) or not (False).
    :vartype include_probabilities: bool
    :ivar include_weighted_entropy: Whether to include the weighted entropy
        in the resulting evaluation (True) or not (False).
    :vartype include_weighted_entropy: bool
    :ivar include_clusters: Whether to include the cluster-wise entropies
        in the resulting evaluation (True) or not (False).
    :vartype include_clusters: bool
    :ivar weight_by_predictions: Whether to compute the weighted entropy
        considering the predictions instead of the reference labels (True) or
        not (False, by default).
    :vartype weight_by_predictions: bool
    :ivar num_clusters: Governs how many clusters must be built when the
        cluster-wise entropies must be computed.
    :vartype num_clusters: int
    :ivar clustering_max_iters: How many iterations are allowed (at most) for
        the cluster algorithm to converge.
    :vartype clustering_max_iters: int
    :ivar clustering_batch_size: How many points consider per batch at each
        iteration of the clustering algoritm. More points imply a more
        accurate clustering. However, they also imply a greater computational
        cost, and thus longer execution time.
    :vartype clustering_batch_size: int
    :ivar clustering_entropy_weights: Whether to use point-wise entropy as
        the sample weights for the clustering (True) or not (False).
    :vartype clustering_entropy_weights: bool
    :ivar clustering_reduce_function: What function use to reduce the entropy
        values in a given cluster to a single one. Either 'mean', 'median',
        'Q1' (first quartile), 'Q3' (third quartile), 'min, or 'max'.
    :vartype clustering_reduce_function: str
    :ivar gaussian_kernel_points: How many points consider to compute the
        gaussian kernel density estimations. Note that this argument has a
        great impact on the time required to generate the plots.
    :vartype gaussian_kernel_points: int
    :ivar report_path: The generated point cloud-like report will be exported
        to the file pointed by the report path.
    :vartype report_path: str
    :ivar plot_path: The generated plots will be stored at the directory
        pointed by the plot path.
    :vartype plot_path: str
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_eval_args(spec):
        """
        Extract the arguments to initialize/instantiate a
        ClassificationUncertaintyEvaluator from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            ClassificationUncertaintyEvaluator.
        """
        # Initialize
        kwargs = {
            'class_names': spec.get('class_names', None),
            'include_probabilities': spec.get('include_probabilities', None),
            'include_weighted_entropy': spec.get(
                'include_weighted_entropy', None
            ),
            'include_clusters': spec.get('include_clusters', None),
            'weight_by_predictions': spec.get('weight_by_predictions', None),
            'num_clusters': spec.get('num_clusters', None),
            'clustering_max_iters': spec.get('clustering_max_iters', None),
            'clustering_batch_size': spec.get('clustering_batch_size', None),
            'clustering_entropy_weights': spec.get(
                'clustering_entropy_weights', None
            ),
            'clustering_reduce_function': spec.get(
                'clustering_reduce_function', None
            ),
            'gaussian_kernel_points': spec.get(
                'gaussian_kernel_points', None
            ),
            'report_path': spec.get('report_path', None),
            'plot_path': spec.get('plot_path', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a ClassificationUncertaintyEvaluator.

        :param kwargs: The attributes for the
            ClassificationUncertaintyEvalutor.
        """
        # Call parent's init
        kwargs['problem_name'] = 'CLASSIFICATION_UNCERTAINTY'
        super().__init__(**kwargs)
        # Assign ClassificationUncertaintyEvaluator attributes
        self.class_names = kwargs.get('class_names', None)
        self.include_probabilities = kwargs.get('include_probabilities', True)
        self.include_weighted_entropy = kwargs.get(
            'include_weighted_entropy', True
        )
        self.include_clusters = kwargs.get('include_clusters', False)
        self.weight_by_predictions = kwargs.get('weight_by_predictions', False)
        self.num_clusters = kwargs.get('num_clusters', False)
        self.clustering_max_iters = int(
            kwargs.get('clustering_max_iters', 128)
        )
        self.clustering_batch_size = int(
            kwargs.get('clustering_batch_size', 4096)
        )
        self.clustering_entropy_weights = kwargs.get(
            'clustering_entropy_weights', None
        )
        self.clustering_reduce_function = kwargs.get(
            'clustering_reduce_function', 'max'
        )
        self.gaussian_kernel_points = kwargs.get(
            'gaussian_kernel_points', 256
        )
        self.report_path = kwargs.get('report_path', None)
        self.plot_path = kwargs.get('plot_path', None)
        # Handle clustering reduce function
        if self.clustering_reduce_function == 'max':
            self.crf = np.max
        elif self.clustering_reduce_function == 'min':
            self.crf = np.min
        elif self.clustering_reduce_function == 'mean':
            self.crf = np.mean
        elif self.clustering_reduce_function == 'median':
            self.crf = np.median
        elif self.clustering_reduce_function == 'Q1':
            self.crf = lambda x: np.quantile(x, 0.25)
        elif self.clustering_reduce_function == 'Q3':
            self.crf = lambda x: np.quantile(x, 0.75)
        else:
            raise EvaluatorException(
                'The given clustering reduce function '
                f'"{self.clustering_reduce_function}" is not supported by '
                'ClassificationUncertaintyEvaluator'
            )

    # ---  EVALUATOR  METHODS  --- #
    # ---------------------------- #
    def eval(self, Zhat, X=None, y=None, yhat=None, F=None):
        r"""
        Evaluate the uncertainty of the given predictions.

        :param Zhat: Predicted class probabilities.
        :type Zhat: :class:`np.ndarray`
        :ivar X: The matrix with the coordinates of the points.
        :vartype X: :class:`np.ndarray`
        :ivar y: The point-wise classes (reference).
        :vartype y: :class:`np.ndarray`
        :ivar yhat: The point-wise classes (predictions).
        :vartype yhat: :class:`np.ndarray`
        :ivar F: The features matrix (it is necessary to compute cluster-wise
            entropies).
        :vartype F: :class:`np.ndarray`
        :return: The evaluation of the classification's uncertainty.
        :rtype: :class:`.ClassificationUncertaintyEvaluation`
        """
        start = time.perf_counter()
        # Adapt Zhat from vector of binary probabilities (no matrix)
        if len(Zhat.shape) < 2:
            Zhat = Zhat.reshape((-1, 1))
            Zhat = np.hstack([1-Zhat, Zhat])
        # Compute point-wise Shannon's entropy
        pwise_entropy = self.compute_pwise_entropy(Zhat)
        # Compute point-wise weighted Shannon's entropy
        weighted_entropy = None
        if self.include_weighted_entropy:
            weighted_entropy = self.compute_weighted_entropy(
                Zhat, y=y, yhat=yhat
            )
        # Compute cluster-wise Shannon's entropies
        cluster_labels, cluster_wise_entropy = None, None
        if self.include_clusters:
            if F is not None:
                cluster_labels, cluster_wise_entropy = \
                    self.compute_cluster_wise_entropy(pwise_entropy, F=F)
            else:
                LOGGING.LOGGER.warning(
                    'ClassificationUncertaintyEvaluator could not compute '
                    'cluster-wise entropy because there are no features '
                    'available (deep learning-based features are currently '
                    'not supported).'
                )
        # Compute point-wise class ambiguity
        class_ambiguity = self.compute_class_ambiguity(Zhat)
        # Log execution time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'ClassificationUncertaintyEvaluator evaluated {Zhat.shape[0]} '
            f'points in {end-start:.3f} seconds.'
        )
        # Return
        return ClassificationUncertaintyEvaluation(
            class_names=self.class_names,
            X=X,
            y=y,
            yhat=yhat,
            Zhat=Zhat if self.include_probabilities else None,
            pwise_entropy=pwise_entropy,
            weighted_entropy=weighted_entropy,
            cluster_labels=cluster_labels,
            cluster_wise_entropy=cluster_wise_entropy,
            class_ambiguity=class_ambiguity,
            gaussian_kernel_points=self.gaussian_kernel_points
        )

    def __call__(self, pcloud, **kwargs):
        """
        Evaluate with extra logic that is convenient for pipeline-based
        execution.

        See :meth:`evaluator.Evaluator.eval`.

        :param pcloud: The point cloud which predicted probabilities must be
            computed to determine the uncertainty measurements.
        :type pcloud: :class:`.PointCloud`
        :param model: The model that computed the predictions.
        :type model: :class:`.Model`
        """
        # Retrieve model
        model = kwargs.get('model', None)
        if model is None:
            raise EvaluatorException(
                'ClassificationUncertaintyEvaluator does not support being '
                'called by a pipeline without model.'
            )

        if not isinstance(model, ClassificationModel):
            raise EvaluatorException(
                'ClassificationUncertaintyEvaluator received a '
                f'"{type(model)}" model which is not a ClassificationModel. '
                'This is not supported.'
            )
        # Determine input type from model
        X = None
        if isinstance(
            model,
            (PointNetPwiseClassifModel, RBFNetPwiseClassifModel)
        ):
            X = pcloud.get_coordinates_matrix()
        else:
            X = pcloud.get_features_matrix(fnames=model.fnames)
        # Obtain predictions and probabilities
        start = time.perf_counter()
        zout = []
        yhat = model._predict(X, zout=zout, plots_and_reports=False)
        Zhat = zout[-1]
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'ClassificationUncertaintyEvaluator computed probabilities for '
            f'{Zhat.shape[0]} points in {end-start:.3f} seconds.'
        )
        # Obtain coordinates
        X = pcloud.get_coordinates_matrix()
        # Obtain classes
        y = kwargs.get('y', None)
        if y is None:
            y = pcloud.get_classes_vector()
        # Obtain features if necessary
        F = None
        if self.include_clusters:
            fnames = getattr(model, "fnames", None)
            if fnames is None:
                fnames = pcloud.get_features_names()
                LOGGING.LOGGER.debug(
                    'ClassificationUncertaintyEvaluator could not derive the '
                    'feature names from the model. '
                    'All available features in the point cloud have been '
                    f'considered:\n{fnames}'
                )
            if fnames is not None and len(fnames) > 0:
                F = pcloud.get_features_matrix(fnames=fnames)
            else:
                LOGGING.LOGGER.warning(
                    'ClassificationUncertaintyEvaluator could not compute '
                    'cluster-based uncertainty because no features were '
                    'available.'
                )
        # Obtain evaluation
        ev = self.eval(Zhat, X=X, y=y, yhat=yhat, F=F)
        out_prefix = kwargs.get('out_prefix', None)
        if ev.can_report() and self.report_path is not None:
            report = ev.report()
            start = time.perf_counter()
            report.to_file(self.report_path, out_prefix=out_prefix)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'The ClassificationUncertaintyEvaluator wrote the point cloud '
                f'in {end-start:.3f} seconds.'
            )
        if ev.can_plot() and self.plot_path is not None:
            start = time.perf_counter()
            ev.plot(path=self.plot_path).plot(out_prefix=out_prefix)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'The ClassificationUncertaintyEvaluator wrote the plots '
                f'in {end-start:.3f} seconds.'
            )

    # ---   UNCERTAINTY QUANTIFICATION METHODS   --- #
    # ---------------------------------------------- #
    def compute_pwise_entropy(self, Zhat):
        r"""
        Compute the point-wise Shannon's entropy for the given predicted
        probabilities.

        Let :math:`\pmb{Z} \in \mathbb{R}^{m \times n_c}` be a matrix
        representing the predicted probabilities for :math:`m` points
        assuming :math:`n_c` classes. The point-wise Shannon entropy for
        point i :math:`e_{i}` can be defined as:

        .. math::

            e_i = - \sum_{j=1}^{n_c}{z_{ij} \log_{2}(z_{ij})}


        :param Zhat: The matrix of point-wise predicted probabilities.
        :type Zhat: :class:`np.ndarray`
        :return: A vector of point-wise Shannon's entropies such that the
            component i is the entropy corresponding to the point i.
        :rtype: :class.`np.ndarray`
        """
        E = -np.sum(Zhat * np.log2(Zhat), axis=1)
        E[np.isnan(E)] = 0  # NaN means no entropy at all (zero)
        return E

    def compute_weighted_entropy(self, Zhat, y=None, yhat=None):
        r"""
        Compute the weighted point-wise Shannon's entropy for the given
        predicted probabilities.

        The weighted Shannon's entropy is the point-wise Shannon's entropy but
        weighting each probability by the frequency of the class with respect
        to some reference labels :math:`\pmb{y} \in \mathbb{Z}^{m}` for
        :math:`n_c` different classes. When the expected reference labels of
        a point cloud (i.e., classification, i.e., self.y) are available, they
        will be considered. Otherwise, when they are not available or the
        ``weight_by_predictions`` flag is true, the predicted labels will be
        considered for the weights.

        The weights can be represented through a vector
        :math:`\pmb{w} \in \mathbb{R}^{n_c}`. Let :math:`m` be the number of
        points and :math:`m_j` be the number of points belonging to class j.
        For then, the components of the weights vector can be defined as:

        .. math::

            w_j = 1 - \dfrac{m_j}{m}

        When using these weights, the less frequent classes will be more
        significant than the more frequent classes. The weighted point-wise
        entropy will be computed as follows:

        .. math::

            e_i = - \sum_{j=1}^{n_c}{w_j z_{ij} \log_{2}(z_{ij})}

        See :meth:`classification_uncertainty_evaluator.ClassificationUncertaintyEvaluator.compute_pwise_entropy`.

        :param Zhat: The matrix of point-wise predicted probabilities.
        :type Zhat: :class:`np.ndarray`
        :return: A vector of weighted point-wise Shannon's entropies such that
            the component i is the entropy corresponding to the point i.
        :rtype: :class:`np.ndarray`
        """
        # Obtain reference
        _y = y
        if _y is None or self.weight_by_predictions:
            _y = yhat
            LOGGING.LOGGER.debug(
                'ClassificationUncertaintyEvaluator considered predictions '
                'for weighted entropy.'
            )
        if _y is None:
            return None
        # Compute weights
        mmax = len(_y)
        m = np.array([
            np.count_nonzero(_y == ci) for ci in range(len(self.class_names))
        ])
        w = 1-m/mmax  # The class-wise weights
        # Compute weighted entropy
        E = -np.sum(w * Zhat * np.log2(Zhat), axis=1)
        E[np.isnan(E)] = 0  # NaN means no entropy at all (zero)
        return E

    def compute_cluster_wise_entropy(self, E, F=None):
        r"""
        Compute the cluster-wise Shannon's entropy for the given predicted
        point-wise entropies and features.

        A KMeans is computed on batches with ``self.clustering_batch_size``
        points up to a maximum of ``self.clustering_max_iters`` iterations
        to extract ``self.num_clusters`` clusters on the feature space. If
        ``self.clustering_entropy_weights`` is True, then the KMeans will scale
        the contribution of each point considering its associated point-wise
        entropy. Finally, all the points belonging to the same cluster will
        have the same cluster-wise entropy which is obtained by reducing
        the entropies in the cluster through the ``self.crf`` function.

        :param E: The point-wise Shannon's entropies
            :math:`\pmb{E} \in \mathbb{R}^{m \times 1}`.
        :param F: The feature matrix
            :math:`\pmb{F} \in \mathbb{R}^{m \times n_f}`.
        :return: A vector of point-wise cluster labels and a vector of
            cluster-wise Shannon's entropies (one cluster-wise per point).
        :rtype: tuple
        """
        # Validate : F is not None
        if F is None:
            raise EvaluatorException(
                'ClassificationUncertaintyEvaluator does not support cluster-'
                'wise entropy computation without features.'
            )
        # Prepare the clustering
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            max_iter=self.clustering_max_iters,
            batch_size=self.clustering_batch_size,
            compute_labels=False,
            n_init='auto'
        )
        # Compute the clustering
        start = time.perf_counter()
        kmeans.fit(
            F,
            sample_weight=E if self.clustering_entropy_weights else None
        )
        cluster_labels = kmeans.predict(F)  # For each point, the cluster index
        # Extract the clusters
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            'ClassificationUncertaintyEvaluator computed KMeans clustering '
            f'in {end-start:.3f} seconds.\n'
            f'The algorithm ran for {kmeans.n_iter_} iterations and processed '
            f'{kmeans.n_steps_} batches on {kmeans.n_features_in_} features.'
        )
        # Find the max entropy for each cluster
        cluster_E = np.array([
            self.crf(E[cluster_labels == cluster_label])
            for cluster_label in range(kmeans.n_clusters)
        ])
        # Assign cluster's max entropy for each point in the cluster
        cwE = cluster_E[cluster_labels]
        # Return clusters and cluster-wise entropy
        return cluster_labels, cwE

    def compute_class_ambiguity(self, Zhat):
        r"""
        Compute a naive point-wise class ambiguity measurement.

        Let :math:`\pmb{Z} \in \mathbb{R}^{m \times n_c}` be a matrix
        representing the predicted probabilities for :math:`m` points
        assuming :math:`n_c` classes. The point-wise class ambiguity for
        point i :math:`a_{i}` can be defined as:

        .. math::

            a_i = 1 - z^{*}_{i} + z^{**}_{i}

        Where :math:`z^{*}_{i}` is the highest prediction for point i and
        :math:`z^*{**}_{i}` is the second highest prediction for point i.

        :param Zhat: The matrix of point-wise predicted probabilities.
        :type Zhat: :class:`np.ndarray`
        :return: A vector of point-wise class ambiguities such that the
            component i is the class ambiguity corresponding to the point i.
        """
        # Sort probabilities
        Zhat = np.sort(Zhat, axis=1)[:, ::-1]
        # Compute class ambiguity considering the most and second most likely
        return 1.0 - Zhat[:, 0] + Zhat[:, 1]

    # ---  PIPELINE METHODS  --- #
    # -------------------------- #
    def eval_args_from_state(self, state):
        """
        Obtain the arguments to call the DLModelEvaluator from the current
        pipeline's state.

        :param state: The pipeline's state.
        :type state: :class:`.SimplePipelineState`
        :return: The dictionary of arguments for calling
            ClassificationUncertaintyEvaluator
        :rtype: dict
        """
        return {
            'pcloud': state.pcloud,
            'model':    state.model.model if hasattr(state.model, 'model')
                        else None
        }
