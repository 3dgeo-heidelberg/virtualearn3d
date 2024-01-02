# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.random_forest_classification_model import \
    RandomForestClassificationModel
from src.model.deeplearn.point_net_pwise_classif_model import \
    PointNetPwiseClassifModel
from src.model.deeplearn.rbf_net_pwise_classif_model import \
    RBFNetPwiseClassifModel
from src.model.deeplearn.dlrun.receptive_field_pre_processor import \
    ReceptiveFieldPreProcessor
from src.inout.model_io import ModelIO
import src.main.main_logger as LOGGING
import sklearn
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from keras.utils.object_identity import ObjectIdentityDictionary
import numpy as np
import tempfile
import os


# ---   CLASS   --- #
# ----------------- #
class ModelSerializationTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Serialization test that checks the serialization (and deserialization)
    of models.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Model serialization test')
        # Prepare training data
        self.num_points = 256
        self.fnames = ['intensity', 'planarity']
        self.num_features = len(self.fnames)
        self.X = np.random.normal(0, 1, (self.num_points, 3))
        self.F = np.random.normal(0, 1, (self.num_points, self.num_features))
        self.class_names = ['Terrain', 'Vegetation', 'Object']
        self.num_classes = len(self.class_names)
        self.y = np.random.randint(0, self.num_classes, self.num_points)

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run model serialization test.

        :return: True if model serialization works as expected for the test
            cases, False otherwise.
        :rtype bool:
        """
        # Prepare tsts
        status = True
        LOGGING.LOGGER.disabled = True  # Disable logging
        # Run tests
        try:
            status = status and self.test_random_forest_classifier()
            status = status and self.test_point_net_pwise_classifier()
            status = status and self.test_rbf_net_pwise_classifier()
        finally:
            LOGGING.LOGGER.disabled = False  # Restore logging
        # Return status
        return status

    # ---  MANY MODELS TESTS  --- #
    # --------------------------- #
    def test_random_forest_classifier(self):
        """
        Test the serialization of the random forest classification model.

        See :class:`.RandomForestClassificationModel`.

        :return: True if model serialization works as expected, False
            otherwise.
        """
        # Model initialization arguments
        init_args = {
            # Model init args
            'autoval_metrics_names': ['OA', 'P', 'wP', 'MCC'],
            'training_type': "stratified_kfold",
            'random_seed': None,
            'shuffle_points': True,
            'autoval_size': 0.3,
            'num_folds': 4,
            'imputer': None,
            "hyperparameter_tuning": {
                "tuner": "GridSearch",
                "hyperparameters": ["n_estimators", "max_depth", "max_samples"],
                "nthreads": -1,
                "num_folds": 5,
                "pre_dispatch": 8,
                "grid": {
                    "n_estimators": [2, 4, 8, 16],
                    "max_depth": [15, 20, 27],
                    "max_samples": [0.6, 0.8, 0.9]
                },
                "report_path": None
            },
            'fnames': self.fnames,
            'stratkfold_report_path': None,
            'stratkfold_plot_path': None,
            # Classification model init args
            'class_names': self.class_names,
            # Random forest classifier init args
            "model_args": {
                "n_estimators": 4,
                "criterion": "entropy",
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0,
                "max_features": "sqrt",
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "bootstrap": True,
                "oob_score": False,
                "n_jobs": 4,
                "warm_start": False,
                "class_weight": None,
                "ccp_alpha": 0.0,
                "max_samples": 0.8
            }
        }
        rf_original = RandomForestClassificationModel(**init_args)
        rf_original.training(self.X, self.y)
        # Temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            # Serialize model
            tmpfile = os.path.join(tmpdir, 'rf')
            ModelIO.write(rf_original, tmpfile)
            # Deserialize model
            rf_deserial = ModelIO.read(tmpfile)
            # Validate deserialized model
            return self.validate_deserialized_model(rf_original, rf_deserial)

    def test_point_net_pwise_classifier(self):
        """
        Test the serialization of the PointNet point-wise classification model.

        See :class:`.PointNetPwiseClassifModel`.

        :return: True if model serialization works as expected, False
            otherwise.
        """
        # Model initialization arguments
        batch_size = 16
        init_args = {
            # Model init args
            'autoval_metrics_names': ['OA', 'P', 'wP', 'MCC'],
            'training_type': "base",
            'random_seed': None,
            'shuffle_points': True,
            'autoval_size': 0.3,
            'num_folds': 4,
            'imputer': None,
            'fnames': ["AUTO"],
            'stratkfold_report_path': None,
            'stratkfold_plot_path': None,
            # Classification model init args
            'class_names': self.class_names,
            # PointNet pwise classifier init args
            "model_args": {
                "fnames": self.fnames,
                "num_classes": len(self.class_names),
                'class_names': self.class_names,
                "num_pwise_feats": 8,
                "pre_processing": {
                    "pre_processor": "furthest_point_subsampling",
                    "to_unit_sphere": True,
                    "support_strategy": "fps",
                    "support_strategy_num_points": 32,
                    "support_strategy_fast": False,
                    "support_chunk_size": 8,
                    "training_class_distribution": [8, 8, 8],
                    "center_on_pcloud": True,
                    "num_points": 64,
                    "num_encoding_neighbors": 1,
                    "fast": False,
                    "neighborhood": {
                        "type": "Rectangular3D",
                        "radius": 1.5,
                        "separation_factor": 1.5
                    },
                    "nthreads": -1,
                    "training_receptive_fields_distribution_report_path": None,
                    "training_receptive_fields_distribution_plot_path": None,
                    "training_receptive_fields_dir": None,
                    "receptive_fields_distribution_report_path": None,
                    "receptive_fields_distribution_plot_path": None,
                    "receptive_fields_dir": None,
                    "training_support_points_report_path": None,
                    "support_points_report_path": None
                },
                "kernel_initializer": "he_normal",
                "pretransf_feats_spec": [
                    {
                        "filters": 4,
                        "name": "prefeats_4A"
                    },
                    {
                        "filters": 4,
                        "name": "prefeats_4B"
                    }
                ],
                "postransf_feats_spec": [
                    {
                        "filters": 4,
                        "name": "posfeats_4"
                    },
                    {
                        "filters": 8,
                        "name": "posfeats_8"
                    },
                    {
                        "filters": 16,
                        "name": "posfeats_end_16"
                    }
                ],
                "tnet_pre_filters_spec": [4, 8],
                "tnet_post_filters_spec": [4, 8],
                "pretransf_feats_F_spec": [
                    {
                        "filters": 4,
                        "name": "prefeats_4A"
                    },
                    {
                        "filters": 4,
                        "name": "prefeats_4B"
                    }
                ],
                "postransf_feats_F_spec": [
                    {
                        "filters": 4,
                        "name": "posfeats_4"
                    },
                    {
                        "filters": 8,
                        "name": "posfeats_8"
                    },
                    {
                        "filters": 16,
                        "name": "posfeats_end_16"
                    }
                ],
                "tnet_pre_filters_F_spec": [4, 8],
                "tnet_post_filters_F_spec": [8, 4],

                "final_shared_mlps": [16, 8, 4],
                "skip_link_features_X": False,
                "include_pretransf_feats_X": False,
                "include_transf_feats_X": True,
                "include_postransf_feats_X": False,
                "include_global_feats_X": True,
                "skip_link_features_F": False,
                "include_pretransf_feats_F": False,
                "include_transf_feats_F": True,
                "include_postransf_feats_F": False,
                "include_global_feats_F": True,
                "_features_structuring_layer": {  # Ignored.
                    # However, initial "_" can be removed to debug
                    "max_radii": [3, 3, 3],
                    "radii_resolution": 3,
                    "angular_resolutions": [1, 2, 4],
                    "concatenation_strategy": "FULL",
                    "dim_out": 8,
                    "trainable_kernel_structure": True,
                    "trainable_kernel_weights": True,
                    "trainable_distance_weights": True,
                    "trainable_feature_weights": True,
                    "batch_normalization": True,
                    "activation": "relu",
                    "freeze_training": True,
                    "freeze_training_init_learning_rate": 1e-3
                },
                "model_handling": {
                    "summary_report_path": None,
                    "training_history_dir": None,
                    "features_structuring_representation_dir": None,
                    "class_weight": [0.8, 0.9, 1],
                    "training_epochs": 3,
                    "batch_size": batch_size,
                    "checkpoint_path": None,
                    "checkpoint_monitor": "loss",
                    "learning_rate_on_plateau": {
                        "monitor": "loss",
                        "mode": "min",
                        "factor": 0.1,
                        "patience": 5000,
                        "cooldown": 5,
                        "min_delta": 0.01,
                        "min_lr": 1e-6
                    },
                    "early_stopping": {
                        "monitor": "loss",
                        "mode": "min",
                        "min_delta": 0.01,
                        "patience": 5000
                    },
                    "fit_verbose": 0
                },
                "compilation_args": {
                    "optimizer": {
                        "algorithm": "SGD",
                        "learning_rate": {
                            "schedule": "exponential_decay",
                            "schedule_args": {
                                "initial_learning_rate": 1e-2,
                                "decay_steps": 1000,
                                "decay_rate": 0.96,
                                "staircase": False
                            }
                        }
                    },
                    "loss": {
                        "function": "class_weighted_binary_crossentropy"
                    },
                    "metrics": [
                        "binary_accuracy",
                        "precision",
                        "recall"
                    ]
                },
                "architecture_graph_path": None,
                "architecture_graph_args": {
                    "show_shapes": True,
                    "show_dtype": True,
                    "show_layer_names": True,
                    "rankdir": "TB",
                    "expand_nested": True,
                    "dpi": 300,
                    "show_layer_activations": True
                }
            },
        }
        pnet_original = PointNetPwiseClassifModel(**init_args)
        pnet_original.training([self.X, self.F], self.y)
        # Temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            # Serialize model
            tmpfile = os.path.join(tmpdir, 'pnet')
            ModelIO.write(pnet_original, tmpfile)
            # Deserialize model
            pnet_deserial = ModelIO.read(tmpfile)
            # Validate deserialized model
            return self.validate_deserialized_model(
                pnet_original, pnet_deserial
            )

    def test_rbf_net_pwise_classifier(self):
        """
        Test the serialization of the RBFNet point-wise classification model.

        See :class:`.RBFNetPwiseClassificationModel`.

        :return: True if model serialization works as expected, False
            otherwise.
        """
        # TODO Rethink : Implement
        return True


    # ---  MODEL VALIDATION METHODS  --- #
    # ---------------------------------- #
    def validate_deserialized_model(self, original, deserial):
        """
        Check that the attributes of the deserialized model (deserial) match
        those of the model before the serialization.

        :param original: The original model, i.e., before serialization.
        :param deserial: The deserialized model.
        :return: True if the deserialized model is valid, False otherwise.
        """
        if original is None:  # No original model was given
            return False
        if deserial is None:  # No deserialized model was given
            return False
        # Recursively validate given models
        return ModelSerializationTest.recursive_object_validation(
            original, deserial
        )

    @staticmethod
    def recursive_object_validation(oriobj, desobj):
        """
        Recursively compare given objects in an attribute-wise way.

        :param oriobj: The original object.
        :type oriobj: object
        :param desobj: The deserialized object.
        :type desobj: object
        :return: True if the objects match (i.e., are equal), false otherwise.
        """
        # Assume certain classes that cannot be inspected are okay
        if isinstance(oriobj, (  # Tuple of classes to assume as okay
            sklearn.tree._tree.Tree,
            ObjectIdentityDictionary
        )):
            return True
        # Prepare validation
        IGNORE_TYPES = (
            int,
            float,
            str,
            bool,
            bytes,
            list,
            tuple,
            dict,
            set,
            np.ndarray,
            tf.Tensor,
            tf.Variable,
            EagerTensor
        )
        TENSOR_TYPES = (
            tf.Tensor,
            tf.Variable,
            EagerTensor
        )
        orivars, desvars = oriobj.__dict__, desobj.__dict__
        for okey in orivars.keys():  # Iterate for each okey (original key)
            # Skip irrelevant attributes (not relevant for serialization)
            if ModelSerializationTest.is_irrelevant_attribute(oriobj, okey):
                continue
            # Check original attribute exists in deserial
            if okey not in desvars:
                return False
            # Check original attribute value matches deserial
            orival, desval = orivars[okey], desvars[okey]
            oritype, destype = type(orival), type(desval)
            if oritype != destype:  # Check types match
                return False
            elif (orival is None) ^ (desval is None):  # Check only one is None
                return False
            elif isinstance(orival, np.ndarray):  # Handle array comparison
                if np.any(orival != desval):  # Element-wise comparison
                    return False
            elif isinstance(orival, TENSOR_TYPES):  # Handle tensor comparison
                if bool(tf.reduce_any(orival != desval)):  # Element-wise comp.
                    return False
            elif (  # Handle list or tuple of TF elements
                    isinstance(orival, (list, tuple)) and
                    len(orival) > 0 and
                    isinstance(orival[0], TENSOR_TYPES)
            ):
                for oritensor, destensor in zip(orival, desval):
                    if len(oritensor.shape) == len(destensor.shape) == 0:
                        continue  # Avoid comparing empty tensors (problematic)
                    if bool(tf.reduce_any(oritensor != destensor)):
                        return False  # Element-wise comparison
            elif orival != desval:  # When they don't match
                if not isinstance(orival, IGNORE_TYPES):  # Recurs. obj. val.
                    valid = ModelSerializationTest.recursive_object_validation(
                        orival, desval
                    )
                    if not valid:
                        return False
                elif isinstance(orival, (list, tuple)):  # Element-wise val.
                    if len(orival) != len(desval):  # Check num elems.
                        return False
                    for orielem, deselem in zip(orival, desval):
                        valid = ModelSerializationTest.recursive_object_validation(
                            orielem, deselem
                        )
                        if not valid:
                            return False
                else:  # Values do not match
                    return False
        # Return True because all previous checks were passed
        return True

    @staticmethod
    def is_irrelevant_attribute(obj, key):
        """
        Determine whether the attribute of a given object is a relevant
        attribute (from the serialization's perspective) or not.

        :param key: The key representing the attribute.
        :type key: str
        :return: True if the attribute is a relevant attribute, False
            otherwise.
        :rtype: bool
        """
        # Ignored attributes for ReceptiveFieldPreProcessor
        if isinstance(obj, ReceptiveFieldPreProcessor):
            return key in [
                'last_call_receptive_fields',
                'last_call_neighborhoods'
            ]
        # Ignore attributes for whatever class
        return key in [
            '_eager_losses',
            '_metrics_lock',
            '_auto_config',
            '_init_input_shape',
            '_init_dtype',
            '_init_sparse',
            '_init_ragged',
            '_inbound_nodes_value',
            '_obj_reference_counts_dict',
            '_auto_get_config',
        ]

