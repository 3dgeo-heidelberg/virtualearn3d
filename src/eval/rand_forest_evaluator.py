# ---   IMPORTS   --- #
# ------------------- #
from src.eval.evaluator import Evaluator, EvaluatorException
from src.eval.rand_forest_evaluation import RandForestEvaluation
from src.utils.dict_utils import DictUtils
from src.main.main_config import VL3DCFG
from sklearn.inspection import permutation_importance
import random


# ---   CLASS   --- #
# ----------------- #
class RandForestEvaluator(Evaluator):
    """
    :author: Alberto M. Esmoris Pena

    Class to evaluate trained random forest models. See
    :class:`.RandomForestClassificationModel`.

    :ivar num_decision_trees: How many estimators consider when plotting the
        decision trees. Zero means none at all, n means consider n decision
        trees, and -1 means consider all the decision trees.
    :vartype num_decision_trees: int
    :ivar compute_permutation_importance: Whether to also compute the
        permutation importance (True) or not (False).
     """

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a RandForestEvaluator.

        :param kwargs: The attributes for the RandForestEvaluator.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Set defaults from VL3DCFG
        kwargs = DictUtils.add_defaults(
            kwargs,
            VL3DCFG['EVAL']['RandForestEvaluator']
        )
        # Initialize attributes of RandForestEvaluator
        self.num_decision_trees = kwargs.get('num_decision_trees', 0)
        self.compute_permutation_importance = kwargs.get(
            'compute_permutation_importance', True
        )
        self.max_tree_depth = kwargs.get('max_tree_depth', 5)

    # ---  EVALUATOR METHODS  --- #
    # --------------------------- #
    def eval(self, model, X=None, y=None, **kwargs):
        """
        Evaluate a trained random forest model.

        :param model: The random forest model.
        :type model: :class:`.RandomForestClassificationModel`
        :param X: The matrix of point-wise features. Rows are points, columns
            are features.
        :param y: The vector of classes. The component i represents the class
            for the point i.
        :return: The evaluation of the trained random forest.
        :rtype: :class:`.RandForestEvaluation`
        """
        if model is None:
            raise EvaluatorException(
                 'RandForestEvaluator cannot evaluate without an input model. '
                 'None was given.'
             )
        # Evaluate : feature permutation importance
        pimportance = None
        if self.compute_permutation_importance:
            pimportance = permutation_importance(model.model, X, y)
        # Evaluate : select decision trees
        trees = None
        if self.num_decision_trees == -1:
            trees = model.model.estimators_
        elif 0 < self.num_decision_trees <= len(model.model.estimators_):
            trees = list(model.model.estimators_)
            random.shuffle(trees)
            trees = trees[:self.num_decision_trees]
        if self.num_decision_trees > len(model.model.estimators_):
            raise EvaluatorException(
                'RandForestEvaluator cannot plot more decision trees than '
                'available. Requested trees are '
                f'{self.num_decision_trees}, available trees are '
                f'{len(model.model.estimators_)}.'
            )
        # Evaluate : return
        pimportance_mean, pimportance_stdev = None, None
        if pimportance is not None:
            pimportance_mean = pimportance['importances_mean']
            pimportance_stdev = pimportance['importances_std']
        return RandForestEvaluation(
            fnames=model.fnames,
            importance=model.model.feature_importances_,
            permutation_importance_mean=pimportance_mean,
            permutation_importance_stdev=pimportance_stdev,
            trees=trees,
            max_tree_depth=self.max_tree_depth
        )
