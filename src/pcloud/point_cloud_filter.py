# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
from src.pcloud.point_cloud import PointCloudException
import src.main.main_logger as LOGGING
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class PointCloudFilter:
    """
    :author: Alberto M. Esmoris Pena

    The base PointCloudFilter class represents a set of operations that can be
    seen as a data filter to be applied on the point cloud. Typically, filters
    are relationals that decide whether

    :ivar conditions: The conditions to be applied by the filter.
    :vartype conditions: list of dict
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, conditions):
        """
        Initialize a PointCloudFilter from a list of conditions.

        :param conditions: The list of conditions. Each condition must be a
            dictionary specifying an operation on the point cloud.
        :type conditions: dict
        """
        self.conditions = conditions

    # ---   FILTER METHODS  --- #
    # ------------------------- #
    def filter(self, pcloud):
        """
        Apply the filter to the given point cloud.

        :param pcloud: The point cloud to be filtered.
        :type pcloud: :class:`.PointCloud`
        :return: The filtered point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Nothing to filter if no conditions are given
        if self.conditions is None:
            return pcloud
        # Apply the conditions to the point cloud
        start = time.perf_counter()
        initial_size = pcloud.get_num_points()
        for condition_idx, condition in enumerate(self.conditions):
            pcloud = self._filter(pcloud, condition, condition_idx)
        filtered_size = pcloud.get_num_points()
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'PointCloudFilter received {initial_size} points and filtered '
            f'out {initial_size-filtered_size} leading to a filtered point '
            f'cloud of {filtered_size} points.'
        )
        # Return filtered point cloud
        return pcloud

    def _filter(self, pcloud, condition, condition_idx):
        """
        Apply a given condition, i.e., filter step.

        :param pcloud: The point cloud to be filtered.
        :param condition: The condition to be applied.
        :type condition: dict
        :param condition_idx: The integer index identifying the condition.
        :return: The filtered point cloud.
        :rtype: :class:`.PointCloud`
        """
        cond_type = condition['condition_type']
        try:
            cond_type_low = cond_type.lower()
            if cond_type_low == 'not_equals':
                return self.apply_relational(
                    pcloud,
                    condition,
                    lambda x, target: x != target
                )
            elif cond_type_low == 'equals':
                return self.apply_relational(
                    pcloud,
                    condition,
                    lambda x, target: x == target
                )
            elif cond_type_low == 'less_than':
                return self.apply_relational(
                    pcloud,
                    condition,
                    lambda x, target: x < target
                )
            elif cond_type_low == 'less_than_or_equal_to':
                return self.apply_relational(
                    pcloud,
                    condition,
                    lambda x, target: x <= target
                )
            elif cond_type_low == 'greater_than':
                return self.apply_relational(
                    pcloud,
                    condition,
                    lambda x, target: x > target
                )
            elif cond_type_low == 'greater_than_or_equal_to':
                return self.apply_relational(
                    pcloud,
                    condition,
                    lambda x, target: x >= target
                )
            elif cond_type_low == 'in':
                return self.apply_relational(
                    pcloud,
                    condition,
                    lambda x, target: [xi in target for xi in x]
                )
            elif cond_type_low == 'not_in':
                return self.apply_relational(
                    pcloud,
                    condition,
                    lambda x, target: [xi not in target for xi in x]
                )
            else:
                LOGGING.LOGGER.error(
                    'PointCloudFilter._filter received an unexpected '
                    f'condition type: "{cond_type}"'
                )
        except Exception as ex:
            LOGGING.LOGGER.error(
                f'PointCloudFilter failed to apply condition {condition_idx} '
                f'with type "{cond_type}".'
            )
            raise PointCloudException(
                'PointCloudFilter._filter failed to apply condition '
                f'{condition_idx} with type "{cond_type}".'
            ) from ex
        LOGGING.LOGGER.warning(
            'PointCloudFilter._filter reached an unexpected execution point.'
        )

    # ---  FILTER UTIL METHODS   --- #
    # ------------------------------ #
    def apply_relational(self, pcloud, condition, relational):
        """
        Apply a relational to filter the point cloud with respect to a given
        condition.

        :param pcloud: The point cloud to be filtered.
        :type pcloud: :class:`.PointCloud`
        :param condition: The condition to be applied as part of the filter.
        :type condition: dict
        :param relational: The function that evaluates the relational.
        :type relational: function
        :return: The filtered point cloud.
        :rtype: :class:`.PointCloud`
        """
        # Prepare relational
        field = condition['value_name']
        field_low = field.lower()
        target = condition['value_target']
        action = condition['action']
        action_low = action.lower()
        # Extract requested value
        x = None
        if field_low == 'classification':
            x = pcloud.get_classes_vector()
        else:
            x = pcloud.get_features_matrix([field]).flatten()
        # Evaluate the relational
        rel_eval = relational(x, target)
        # Apply action
        if action_low == 'preserve':
            return pcloud.preserve_mask(rel_eval)
        elif action_low == 'discard':
            return pcloud.remove_mask(rel_eval)
        else:
            raise PointCloudException(
                'PointCloudFilter.apply_relational received an unexpected '
                f'action: "{action}".'
            )



