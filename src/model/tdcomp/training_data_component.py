# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.main.vl3d_exception import VL3DException
from src.utils.dict_utils import DictUtils
import src.main.main_logger as LOGGING


# ---   EXCEPTIONS   --- #
# ---------------------- #
class TrainingDataComponentException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to components for training data pipelines.
    See :class:`.Model`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class TrainingDataComponent:
    """
    :author: Alberto M. Esmoris Pena

    Abstract class providing the interface governing any training data
    component.

    :ivar component_args: The key-word arguemnts for the component of the
        training data pipeline.
    :vartype component_args: dict
    """
    # ---  SPECIFICATION ARGUMENTS  --- #
    # ⨪-------------------------------- #
    @staticmethod
    def extract_component_args(spec):
        """
        Extract the arguments to initialize/instantiate a TrainingDataComponent
        from a key-word specification.

        :param spec: The key-word specification containing the arguments.
        :return: The arguments to initialize/instantiate a
            TrainingDataComponent.
        """
        # Initialize
        kwargs = {
            'component_args': spec.get('component_args', None)
        }
        # Delete keys with None value
        kwargs = DictUtils.delete_by_val(kwargs, None)
        # Return
        return kwargs

    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Root initialization for any instance of type TrainingDataComponent.

        :param kwargs: The attributes for the TrainingDataComponent.
        """
        # Fundamental initialization of any training data component
        self.component_args = kwargs.get('component_args', None)

    # ---   CALL   --- #
    # ---------------- #
    @abstractmethod
    def __call__(self, X, y):
        """
        Run the component on the given training data and return the transformed
        version of the training data.

        :param X: The input to the model, typically the attributes.
        :param y: The expected classes.
        :return: The new attributes and expected classes for model training.
        :rtype: tuple (X, y)
        """
        pass

    # ---   PIPELINE BUILDING   --- #
    # ----------------------------- #
    @staticmethod
    def build_pipeline(spec):
        """
        Build a pipeline from a training data pipeline specification.

        :param spec: A list of dictionaries where each dictionary represents
            a training data component.
        :type spec: list
        :return: The built pipeline.
        :rtype: list of :class:`.TrainingDataComponent`
        """
        if spec is None:
            LOGGING.LOGGER.error(
                'TrainingDataComponent.build_pipeline received a None '
                'specification.'
            )
            raise TrainingDataComponentException(
                'TrainingDataComponent.build_pipeline received a None '
                'specification.'
            )
        if not isinstance(spec, list):
            LOGGING.LOGGER.error(
                'TrainingDataComponent.build_pipeline did NOT receive a list '
                'of components.'
            )
            raise TrainingDataComponentException(
                'TrainingDataComponent.build_pipeline did NOT receive a list '
                'of components.'
            )
        return [
            TrainingDataComponent.build_component(component_spec)
            for component_spec in spec
        ]

    @staticmethod
    def build_component(spec):
        """
        Build the received training data component.

        :param spec: The specification of the training data component to be
            built.
        :return: The built training data component.
        :rtype: :class:`.TrainingDataComponent`
        """
        # Handle component type
        comp_name = spec.get('component', None)
        if comp_name is None or not isinstance(comp_name, str):
            LOGGING.LOGGER.error(
                f'Specified training data component "{comp_name}" is wrong. '
                f'It should be a string but it is "{type(comp_name)}".'
            )
            raise TrainingDataComponentException(
                f'Specified training data component "{comp_name}" is wrong. '
                f'It should be a string but it is "{type(comp_name)}".'
            )
        comp_low = comp_name.lower()
        # Build component
        if comp_low == 'smote':  # SMOTE
            from src.model.tdcomp.smote import SMOTE
            return SMOTE(**spec)
        elif comp_low == 'classwisesampler':  # ClasswiseSampler
            from src.model.tdcomp.classwise_sampler import ClasswiseSampler
            return ClasswiseSampler(**spec)
        else:  # Unexpected component received
            LOGGING.LOGGER.error(
                f'Unexpected training data component: "{comp_name}".'
            )
            raise TrainingDataComponentException(
                f'Unexpected training data component: "{comp_name}".'
            )



