from src.eval.classification_evaluator import ClassificationEvaluator
from src.eval.classification_uncertainty_evaluator import \
    ClassificationUncertaintyEvaluator
from src.eval.raster_grid_evaluator import RasterGridEvaluator
from src.eval.deeplearn.dl_model_evaluator import DLModelEvaluator


# ---   CLASS   --- #
# ----------------- #
class MainEval:
    """
    :author: Alberto M. Esmoris Pena

    Class handling the entry point for evaluation tasks
    """
    # ---  MAIN METHOD   --- #
    # ---------------------- #
    @staticmethod
    def main(spec):
        """
        Entry point logic for evaluation tasks

        :param spec: Key-word specification
        """
        # TODO Rethink : Implement
        print('main_vl3d_eval')  # TODO Remove
        pass

    # ---  EXTRACT FROM SPEC  --- #
    # --------------------------- #
    @staticmethod
    def extract_eval_class(spec):
        """
        Extract the evaluator's class from the key-word specification.

        :param spec: The key-word specification.
        :return: Class representing/realizing an evaluator.
        :rtype: :class:`.Evaluator`
        """
        eval = spec.get('eval', None)
        if eval is None:
            raise ValueError(
                'Evaluation requires an evaluator. None was specified.'
            )
        # Check evaluator class
        eval_low = eval.lower()
        if eval_low == 'classificationevaluator':
            return ClassificationEvaluator
        elif eval_low == 'classificationuncertaintyevaluator':
            return ClassificationUncertaintyEvaluator
        elif eval_low == 'rastergridevaluator':
            return RasterGridEvaluator
        elif eval_low == 'dlmodelevaluator':
            return DLModelEvaluator
        # An unknown evaluator was specified
        return ValueError(f'There is no known evaluator "{eval}"')
