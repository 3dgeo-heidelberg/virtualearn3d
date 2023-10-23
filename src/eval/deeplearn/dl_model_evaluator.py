# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluator import Evaluator, EvaluatorException
from src.eval.deeplearn.dl_model_evaluation import DLModelEvaluation
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING
import time


# ---   CLASS   --- #
# ----------------- #
class DLModelEvaluator(Evaluator):
    """
    :author: Alberto M. Esmoris Pena

    Class to evaluate deep learning models.

    :ivar dlmodel: The deep learning model to be evaluated
    :vartype dlmodel: :class:`.Model`
    :ivar pwise_output_path: Where to export the point-wise output.
    :vartype pwise_output_path: str
    :ivar pwise_activations_path: Where to export the point-wise activations.
    :vartype pwise_activations_path: str
    :ivar accept_pipeline_state_predictions: Whether to accept predictions
        from a pipeline's state (True) or not (False).
    :vartype accept_pipeline_state_predictions: bool
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_eval_args(spec):
        """
        Extract the arguments to initialize/instantiate a DLModelEValuator
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a DLModelEvaluator.
        """
        # Initialize
        kwargs = {
            'pointwise_model_output_path':
                spec.get('pointwise_model_output_path', None),
            'pointwise_model_activations_path':
                spec.get('pointwise_model_activations_path', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a DLModelEvaluator.

        :param kwargs: The attributes for the DLModelEvaluator.
        """
        # Call parent's init
        kwargs['problem_name'] = 'DL_MODEL'
        super().__init__(**kwargs)
        # Assign DLModelEvaluator attributes
        self.dlmodel = kwargs.get('dlmodel', None)
        self.pwise_output_path = kwargs.get(
            'pointwise_model_output_path', None
        )
        self.pwise_activations_path = kwargs.get(
            'pointwise_model_activations_path', None
        )
        self.accept_pipeline_state_predictions = kwargs.get(
            'accept_pipeline_state_predictions', False
        )

    # ---  EVALUATOR METHODS   --- #
    # ---------------------------- #
    def eval(self, X, y=None, **kwargs):
        """
        Evaluate the DL model.

        Potential evaluations are the point-wise outputs of the model and
        the point-wise activations of a hidden layer.

        :param X: Input data for the evaluation.
        :param y: Expected values for the evaluation.
        :return: The evaluation of the deep learning model.
        :rtype: :class:`.DLModelEvaluation`
        """
        # Validate
        if self.dlmodel is None:
            raise EvaluatorException(
                'DLModelEvaluator needs a deep learning model to evaluate. '
                'None was given.'
            )
        # Start time (ignores validation)
        start = time.perf_counter()
        # Evaluate : Point-wise outputs
        yhat, zhat = None, None
        if self.pwise_output_path:
            local_start = time.perf_counter()
            zhat = []
            yhat = self.dlmodel._predict(X, F=None, y=y, zout=zhat)
            zhat = zhat[-1]
            local_end = time.perf_counter()
            LOGGING.LOGGER.info(
                'DLModelEvaluator computed point-wise outputs in '
                f'{local_end-local_start:.3f} seconds.'
            )
        # Evaluate : Point-wise activations
        activations = None
        if self.pwise_activations_path:
            local_start = time.perf_counter()
            activations = self.dlmodel.compute_pwise_activations(X)
            local_end = time.perf_counter()
            LOGGING.LOGGER.info(
                'DLModelEvaluator computed point-wise activations in '
                f'{local_end-local_start:.3f} seconds.'
            )
        # Log execution time
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'DLModelEvaluator evaluated {len(X)} points in '
            f'{end-start:.3f} seconds.'
        )
        # Return
        return DLModelEvaluation(
            X=X,
            y=y,
            yhat=yhat,
            zhat=zhat,
            activations=activations,
            class_names=getattr(self.dlmodel, 'class_names', None)
        )

    def __call__(self, x, **kwargs):
        """
        Evaluate with extra logic that is convenient for pipeline-based
        execution.

        See :meth:`evaluator.Evaluator.eval`.
        """
        # Obtain evaluation
        ev = self.eval(x, **kwargs)
        out_prefix = kwargs.get('out_prefix', None)
        if ev.can_report():
            pwise_output_path = kwargs.get(
                "pwise_output_path", self.pwise_output_path
            )
            pwise_activations_path = kwargs.get(
                'pwise_activations_path', self.pwise_activations_path
            )
            if(
                pwise_output_path is not None or
                pwise_activations_path is not None
            ):
                start = time.perf_counter()
                report = ev.report()
                report.to_file(
                    pwise_output_path,
                    pwise_activations_path=pwise_activations_path,
                    out_prefix=out_prefix
                )
                end = time.perf_counter()
                LOGGING.LOGGER.info(
                    f'DLModel reports written in {end-start:.3f} seconds.'
                )

    # ---  PIPELINE METHODS  --- #
    # -------------------------- #
    def lazy_prepare(self, state):
        """
        Prepare the DLModelEvaluator, so it can take the dlmodel from the
        pipeline's state that was not available when instantiating the
        evaluator.

        :param state: The pipeline's state.
        :type state: :class:`.SimplePipelineState`
        :return: Nothing, but the internal state of the DLModelEvaluator is
            updated.
        """
        self.dlmodel = state.model

    def eval_args_from_state(self, state):
        """
        Obtain the arguments to call the DLModelEvaluator from the current
        pipeline's state .

        :param state: The pipeline's state
        :type state: :class:`.SimplePipelineState`
        :return: The dictionary of arguments for calling DLModelEvaluator
        :rtype: dict
        """
        return {
            'x': state.pcloud.get_coordinates_matrix()
        }
