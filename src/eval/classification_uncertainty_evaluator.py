# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluator import Evaluator, EvaluatorException
from src.eval.classification_uncertainty_evaluation import \
    ClassificationUncertaintyEvaluation
from src.model.classification_model import ClassificationModel
from src.model.deeplearn.point_net_pwise_classif_model import \
    PointNetPwiseClassifModel
import src.main.main_logger as LOGGING
from src.utils.dict_utils import DictUtils
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
        self.report_path = kwargs.get('report_path', None)
        self.plot_path = kwargs.get('plot_path', None)

    # ---  EVALUATOR  METHODS  --- #
    # ---------------------------- #
    def eval(self, Zhat, X=None, y=None, yhat=None):
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
        :return: The evaluation of the classification's uncertainty.
        :rtype: :class:`.ClassificationUncertaintyEvaluation`
        """
        start = time.perf_counter()
        # Compute point-wise Shannon's entropy
        pwise_entropy = self.compute_pwise_entropy(Zhat)
        # Compute point-wise Shannon's entropy on feature half-spaces
        weighted_hspace_entropy = self.compute_weighted_hspace_entropy(Zhat)
        # Compute point-wise class ambiguity
        class_ambiguity = self.compute_class_ambiguity(Zhat)
        # Log execution time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'ClassificationUncertaintyEvaluator evaluated {Zhat.shape[0]}'
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
            weighted_hspace_entropy=weighted_hspace_entropy,
            class_ambiguity=class_ambiguity
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
        if isinstance(model, PointNetPwiseClassifModel):
            X = pcloud.get_coordinates_matrix()
        else:
            X = pcloud.get_features_matrix(fnames=model.fnames)
        # Obtain predictions and probabilities
        start = time.perf_counter()
        zout = []
        yhat = model._predict(X, zout=zout)
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
        # Obtain evaluation
        ev = self.eval(Zhat, X=X, y=y, yhat=yhat)
        out_prefix = kwargs.get('out_prefix', None)
        if ev.can_report():
            report = ev.report()
            start = time.perf_counter()
            report.to_file(self.report_path, out_prefix=out_prefix)
            end = time.perf_counter()
            LOGGING.LOGGER.info(
                'The ClassificationUncertaintyEvaluator wrote the point cloud '
                f'in {end-start:.3f} seconds.'
            )
        if ev.can_plot():
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
        Compute the point-wise Shannon's entropy for the given predictions.

        Let :math:`\pmb{Z} \in \mathbb{R}^{m \times n_c}` be a matrix
        representing the predicted probabilities for :math:`m` points
        assuming :math:`n_c` classes. The point-wise Shannon entropy for
        point i :math:`e_{i}` can be defined as:

        .. math::

            e_i = - \sum_{j=1}^{n_c} z_{ij} \log_{2}(z_{ij})


        :param Zhat: The matrix of point-wise predicted probabilities.
        :type Zhat: :class:`np.ndarray`
        :return: A vector of point-wise Shannon's entropy such that the
            component i is the entropy corresponding to the point i.
        """
        return -np.sum(Zhat * np.log2(Zhat), axis=1)

    def compute_weighted_hspace_entropy(self, Zhat):
        # TODO Rethink : Implement
        return np.zeros(Zhat.shape[0])  # TODO Remove : Just a placeholder

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
