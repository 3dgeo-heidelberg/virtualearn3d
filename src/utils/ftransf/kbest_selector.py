# ---   IMPORTS   --- #
# ------------------- #
from src.utils.ftransf.feature_transformer import FeatureTransformer, \
    FeatureTransformerException
from src.utils.dict_utils import DictUtils
from src.report.best_score_selection_report import BestScoreSelectionReport
import src.main.main_logger as LOGGING
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression
import time


# ---   CLASS   --- #
# ----------------- #
class KBestSelector(FeatureTransformer):
    """
    :author: Alberto M. Esmoris Pena

    Class for transforming features by preserving only the top k features
    for a particular task.

    :ivar k: The number of best features to select.
    :vartype k: int
    :ivar scoref: The score function f(F, y) to evaluate the features F
        to predict the values of y.
    :vartype scoref: callable
    :ivar score_name: The name of the score used for the evaluations.
    :vartype score_name: str
    """

    # ---  SPECIFICATION ARGUMENTS  --- #
    # --------------------------------- #
    @staticmethod
    def extract_ftransf_args(spec):
        """
        Extract the arguments to initialize/instantiate a KBestSelector.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a KBestSelector.
        """
        # Initialize from parent
        kwargs = FeatureTransformer.extract_ftransf_args(spec)
        # Extract particular arguments of KBestSelector
        kwargs['k'] = spec.get('k', None)
        # Extract score function from type
        valid_score = KBestSelector.handle_score_from_type(spec, kwargs)
        if not valid_score:
            raise ValueError(
                'The KBestSelector specification contains an invalid type: '
                f'"{type}"'
            )
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    @staticmethod
    def handle_score_from_type(spec, kwargs):
        """
        Handle the score from the specified type.

        :param spec: The specification.
        :param kwargs: The key-word arguments being built by
            :meth:`kbest_selector.KBestSelector.extract_ftrasnf_args` to
            initialize a KBestSelector.
        :return: True if a valid score was obtained from the given type,
            False otherwise.
        """
        _type = spec.get('type', None)
        valid_score = _type is not None
        if valid_score:  # Score function f(X, y)
            type_low = _type.lower()
            if type_low == 'classification':
                kwargs['scoref'] = f_classif
                kwargs['score_name'] = 'F-value'
            elif type_low == 'regression':
                kwargs['scoref'] = f_regression
                kwargs['score_name'] = 'F-value'
            else:
                valid_score = False
        return valid_score

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize/instantiate a KBestSelector.

        :param kwargs: The attributes for the KBestSelector.
        """
        # Call parent init
        super().__init__(**kwargs)
        # Assign attributes
        self.k = kwargs.get('k', None)
        if self.k is None:
            raise FeatureTransformerException(
                'KBestSelector requires k (number of selected features) is '
                'specified. None was given.'
            )
        self.scoref = kwargs.get('scoref', None)
        if self.scoref is None:
            raise FeatureTransformerException(
                'KBestSelector requires a valid score function. None was '
                'given.'
            )
        self.score_name = kwargs.get('score_name', 'SCORE')

    # ---  FEATURE TRANSFORM METHODS  --- #
    # ----------------------------------- #
    def transform(self, F, y=None, fnames=None, out_prefix=None):
        """
        The fundamental feature transform logic defining the k-best selector.

        See :class:`.FeatureTransformer` and
        :meth:`feature_transformer.FeatureTransformer.transform`.
        """
        # Check selection is possible
        if y is None:
            raise FeatureTransformerException(
                'KBestSelector cannot transform features without the '
                'corresponding classes.'
            )
        # Transform
        old_num_features = F.shape[1]
        start = time.perf_counter()
        kb = SelectKBest(score_func=self.scoref, k=self.k)
        F = kb.fit_transform(F, y)
        end = time.perf_counter()
        new_num_features = F.shape[1]
        # Register selected features
        self.selected_features = kb.get_support()
        # Report scores
        self.report(
            BestScoreSelectionReport(
                self.fnames if fnames is None else fnames,
                kb.scores_,
                self.score_name,
                pvalues=kb.pvalues_,
                selected_features=self.selected_features
            ),
            out_prefix=out_prefix
        )
        # Log transformation
        LOGGING.LOGGER.info(
            'KBestSelector transformed {n1} features into {n2} features in '
            '{texec:.3f} seconds.'.format(
                n1=old_num_features,
                n2=new_num_features,
                texec=end-start
            )
        )
        # Return
        return F
