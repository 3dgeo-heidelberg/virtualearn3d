# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.random_forest_classification_model import \
    RandomForestClassificationModel
from src.model.deeplearn.point_net_pwise_classif_model import \
    PointNetPwiseClassifModel
from src.model.deeplearn.rbf_net_pwise_classif_model import \
    RBFNetPwiseClassifModel
from src.model.deeplearn.arch.point_net import PointNet
from src.model.deeplearn.arch.rbfnet import RBFNet
from src.model.deeplearn.dlrun.receptive_field_pre_processor import \
    ReceptiveFieldPreProcessor
from src.model.deeplearn.arch.architecture import Architecture
from src.inout.model_io import ModelIO
import src.main.main_logger as LOGGING
import sklearn
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from keras.utils.object_identity import ObjectIdentityDictionary
import numpy as np
import weakref
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
        rf_original.training(self.F, self.y)
        # Temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            # Serialize model
            tmpfile = os.path.join(tmpdir, 'rf')
            ModelIO.write(rf_original, tmpfile)
            # Deserialize model
            rf_deserial = ModelIO.read(tmpfile)
            # Validate deserialized model
            return self.validate_deserialized_model(
                rf_original,
                rf_deserial,
                original_y=rf_original._predict(self.F),
                deserial_y=rf_deserial._predict(self.F)
            )

    def test_point_net_pwise_classifier(self):
        """
        Test the serialization of the PointNet point-wise classification model.

        See :class:`.PointNetPwiseClassifModel`.

        :return: True if model serialization works as expected, False
            otherwise.
        """
        # Model initialization arguments
        batch_size = 4
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
                    "nthreads": 4,
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
                    "fit_verbose": 0,
                    "predict_verbose": 0
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
            pnet_deserial.model = pnet_deserial.model.compile()
            # Validate deserialized model
            return self.validate_deserialized_model(
                pnet_original,
                pnet_deserial,
                original_y=pnet_original.model.predict([self.X, self.F]),
                deserial_y=pnet_deserial.model.predict([self.X, self.F])
            )

    def test_rbf_net_pwise_classifier(self):
        """
        Test the serialization of the RBFNet point-wise classification model.

        See :class:`.RBFNetPwiseClassificationModel`.

        :return: True if model serialization works as expected, False
            otherwise.
        """
        # Model initialization arguments
        batch_size = 4
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
            # RBFNet pwise classifier init args
            "model_args": {
                "fnames": self.fnames,
                "num_classes": len(self.class_names),
                "class_names": self.class_names,
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
                        "type": "sphere",
                        "radius": 1.5,
                        "separation_factor": 1.5
                    },
                    "nthreads": 4,
                    "training_receptive_fields_distribution_report_path": None,
                    "training_receptive_fields_distribution_plot_path": None,
                    "training_receptive_fields_dir": None,
                    "receptive_fields_distribution_report_path": None,
                    "receptive_fields_distribution_plot_path": None,
                    "receptive_fields_dir": None,
                    "training_support_points_report_path": None,
                    "support_points_report_path": None
                },
                "rbfs": [
                    {
                        "max_radii": [1.0, 1.0, 1.0],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 2],
                        "structure_initialization_type": "concentric_grids",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [0.6, 0.6, 0.6],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 2],
                        "structure_initialization_type": "concentric_grids",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [0.3, 0.3, 0.3],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 2],
                        "structure_initialization_type": "concentric_grids",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [1.0, 1.0, 1.0],
                        "radii_resolution": 3,
                        "angular_resolutions": [1, 2, 3],
                        "structure_initialization_type": "concentric_rectangulars",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [1.0, 1.0, 1.0],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 3],
                        "structure_initialization_type": "concentric_cylinders",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [0.6, 0.6, 0.6],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 3],
                        "structure_initialization_type": "concentric_cylinders",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [0.3, 0.3, 0.3],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 3],
                        "structure_initialization_type": "concentric_cylinders",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [1.0, 1.0, 1.0],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 3],
                        "structure_initialization_type": "concentric_ellipsoids",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [0.6, 0.6, 0.6],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 3],
                        "structure_initialization_type": "concentric_ellipsoids",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [0.3, 0.3, 0.3],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 3],
                        "structure_initialization_type": "concentric_ellipsoids",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [1.0, 1.0, 1.0],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 3],
                        "structure_initialization_type": "cone",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [0.6, 0.6, 0.6],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 3],
                        "structure_initialization_type": "cone",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [0.3, 0.3, 0.3],
                        "radii_resolution": 2,
                        "angular_resolutions": [1, 3],
                        "structure_initialization_type": "cone",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    },
                    {
                        "max_radii": [1.0, 1.0, 1.0],
                        "radii_resolution": 1,
                        "angular_resolutions": [1],
                        "structure_initialization_type": "zeros",
                        "trainable_kernel_structure": True,
                        "trainable_kernel_sizes": True,
                        "kernel_function_type": "Gaussian",
                        "batch_normalization": False,
                        "activation": None
                    }
                ],
                "tnet_pre_filters_spec": [4, 8, 16],
                "tnet_post_filters_spec": [32, 16, 8],
                "enhanced_dim": [4, 8, 32],
                "after_features_dim": [32, 16, 8],
                "after_features_type": "Conv1D",
                "include_prepooling_features": True,
                "include_global_features": True,
                "transform_input_features": True,
                "skip_link_features": False,
                "feature_structuring": {
                    "max_radii": [3.0, 3.0, 3.0],
                    "angular_resolutions": [1, 3],
                    "concatenation_strategy": "OPAQUE",
                    "trainable_QX": True,
                    "trainable_QW": True,
                    "trainable_omegaD": True,
                    "trainable_omegaF": True,
                    "enhance": True,
                    "transformed_structure": True,
                    "fsl_input_features_dim_out": "dim_in",
                    "fsl_processed_features_dim_out": "dim_in",
                    "fsl_enhanced_processed_features_dim_out": "dim_in",
                    "fsl_prepool_features_dim_out": "dim_in",
                    "fsl_global_features_dim_out": "dim_in",
                    "fsl_rbf_features_dim_out": "dim_in",
                    "fsl_rbf_enhanced_features_dim_out": "dim_in"
                },
                "feature_processing": {
                    "num_kernels": 8,
                    "a": 0.1,
                    "b": 1,
                    "kernel_function_type": "Gaussian",
                    "trainable_M": True,
                    "trainable_Omega": True,
                    "enhance": True,
                    "batch_normalization": False
                },
                "model_handling": {
                    "summary_report_path": None,
                    "training_history_dir": None,
                    "features_structuring_representation_dir": None,
                    "rbf_feature_extraction_representation_dir": None,
                    "rbf_feature_processing_representation_dir": None,
                    "class_weight": [0.8, 0.9, 1.0],
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
                        "min_delta": 0.001,
                        "patience": 50
                    },
                    "fit_verbose": 0,
                    "predict_verbose": 0
                },
                "compilation_args": {
                    "optimizer": {
                        "algorithm": "SGD",
                        "learning_rate": {
                            "schedule": "exponential_decay",
                            "schedule_args": {
                                "initial_learning_rate": 1e-3,
                                "decay_steps": 33,
                                "decay_rate": 0.96,
                                "staircase": False
                            }
                        }
                    },
                    "loss": {
                        "function": "class_weighted_binary_crossentropy"
                    },
                    "metrics": [
                        "binary_accuracy"
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
        rbfnet_original = RBFNetPwiseClassifModel(**init_args)
        rbfnet_original.training([self.X, self.F], self.y)
        # Temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            # Serialize model
            tmpfile = os.path.join(tmpdir, 'rbfnet')
            ModelIO.write(rbfnet_original, tmpfile)
            # Deserialize model
            rbfnet_deserial = ModelIO.read(tmpfile)
            rbfnet_deserial.model = rbfnet_deserial.model.compile()
            # Validate deserialized model
            return self.validate_deserialized_model(
                rbfnet_original,
                rbfnet_deserial,
                original_y=rbfnet_original.model.predict([self.X, self.F]),
                deserial_y=rbfnet_deserial.model.predict([self.X, self.F])
            )

    # ---  MODEL VALIDATION METHODS  --- #
    # ---------------------------------- #
    def validate_deserialized_model(
        self, original, deserial, original_y=None, deserial_y=None
    ):
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
        if not ModelSerializationTest.recursive_object_validation(
            original, deserial
        ):
            return False
        # Validate the output of given models
        if not ModelSerializationTest.model_output_validation(
            original_y, deserial_y
        ):
            return False
        # All validations were successfully passed
        return True

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
        IGNORE_TYPES = ( # Types that must not be considered as general objects
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
            EagerTensor,
        )
        TENSOR_TYPES = (  # Tensorflow data types
            tf.Tensor,
            tf.Variable,
            EagerTensor
        )
        try:
            orivars, desvars = oriobj.__dict__, desobj.__dict__
        except AttributeError as aerr:
            return True  # Consider objects without attributes as passed
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
            # Extract elements for tuples of single element
            if isinstance(orival, tuple) and len(orival) == 1:
                orival, oritype = orival[0], type(orival[0])
            if isinstance(desval, tuple) and len(desval) == 1:
                desval, destype = desval[0], type(desval[0])
            # Determine whether the attribute is a symbolic keras tensor
            is_keras_tensor = False
            try:
                is_keras_tensor = tf.keras.backend.is_keras_tensor(orival)
            except ValueError as verr:
                pass
            # Determine whether the list or tuple contains keras tensors
            are_keras_tensors = []
            if isinstance(orival, (list, tuple)):
                for i in range(len(orival)):
                    _is_keras_tensor = False
                    try:
                        _is_keras_tensor = tf.keras.backend.is_keras_tensor(
                            orival[i]
                        )
                    except ValueError as verr:
                        pass
                    are_keras_tensors.append(_is_keras_tensor)
            # Handle checks to validate that original matches deserial
            if oritype != destype:  # Check types match
                if(  # Leverage deserial list or tuple to array, if possible
                    isinstance(orival, np.ndarray) and
                    isinstance(desval, (list, tuple))
                ):
                    desval = np.array(desval)
                    destype = type(desval)
                elif(  # Leverage original list or tuple to array, if possible
                    isinstance(desval, np.ndarray) and
                    isinstance(orival, (list, tuple))
                ):
                    orival = np.array(orival)
                    oritype = type(orival)
                else:  # Otherwise, types do not match
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
            elif is_keras_tensor:  # Handle KerasTensor
                if orival.shape != desval.shape or orival.dtype != desval.dtype:
                    return False
            elif len(are_keras_tensors) > 0 and are_keras_tensors[0]:
                # Handle list or tuple of Keras tensors
                for i in range(len(orival)):
                    try:
                        if(
                            orival[i].shape != desval[i].shape or
                            orival[i].dtype != desval[i].dtype
                        ):
                            return False
                    except AttributeError as aerr:
                        try:  # Compare mixed lists (elems. with diff. types.)
                            if orival[i] != desval[i]:
                                return False
                        except Exception:  # Extend exception info.
                            raise AttributeError(
                                f'orival[{i}] type is {type(orival[i])}\n'
                                f'desval[{i}] type is {type(desval[i])}'
                            ) from aerr
            # Handle general comparisons
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
                    try:
                        orival, desval = ModelSerializationTest \
                            .sort_lists_or_tuples(orival, desval)
                        for orielem, deselem in zip(orival, desval):
                            valid = ModelSerializationTest.\
                                recursive_object_validation(orielem, deselem)
                            if not valid:
                                return False
                    except AttributeError as aerr:
                        return False
                elif isinstance(orival, dict):  # Element-wise val. on dicts.
                    for dkey in orival.keys():
                        dorival, ddesval = orival[dkey], desval[dkey]
                        if (
                            not isinstance(dorival, IGNORE_TYPES) or
                            isinstance(dorival, (list, tuple))
                        ):
                            valid = ModelSerializationTest.\
                                recursive_object_validation(dorival, ddesval)
                            if not valid:
                                return False
                        elif dorival != ddesval:
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
            if key in [
                'last_call_receptive_fields',
                'last_call_neighborhoods'
            ]:
                return True
        # Ignore attributes for whatever class
        if key in [
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
            '_keras_inputs_ids_and_indices',
            'flat_input_ids',
            'flat_output_ids',
            '_self_unconditional_checkpoint_dependencies',
            '_self_unconditional_dependency_names',
            '_seed',
            '_kwargs',
            '_steps_per_execution',
            '_train_counter',
            '_nodes_by_depth',
            '_layer_call_argspecs',
            '_tensor_usage_count',
            '_iterations',
            '_current_learning_rate',
            '_index_dict',
            'momentums',
            '_built',
            'compiled_loss',
            'compiled_metrics',
            'train_tf_function',
            'train_function',
            'history',
            'architecture_graph_args',
            'predict_function'
        ]:
            return True
        # Ignore objects
        if type(obj) == tf.keras.layers.Dot:
            return True
        if isinstance(obj, weakref.WeakKeyDictionary):
            return True
        # Ignore attributes for certain objects
        if isinstance(obj, Architecture) and key in [
            "inlayer"
        ]:
            return True
        if isinstance(obj, PointNet) and key in [
            'pretransf_feats_X', 'pretransf_feats_F',
            'postransf_feats_X', 'postransf_feats_F',
            'transf_feats_X', 'transf_feats_F',
            'X', 'F',
            'Xtransf', 'Ftransf'
        ]:
            return True
        if isinstance(obj, RBFNet) and key in [
            'rbf_layers', 'rbf_output_tensors',
            'rbf_feat_proc_layer',
            'X', 'F',
            'Xtransf', 'Ftransf',
            'prepool_feats_tensor',
            'global_feats_tensor',
            'feature_processing_tensor',
            'feature_processing_layer'
        ]:
            return True
        # Irrelevancy checks were passed
        return False

    @staticmethod
    def sort_lists_or_tuples(x, y):
        """
        Generate a sorted version of given lists or tuples.

        :param x: First input list or tuple.
        :type x: list or tuple
        :param y: Second input list or tuple.
        :type y: list or tuple
        :return: Sorted version of the input lists or tuples.
        :rtype: list
        """
        # Sort elements if they are layers
        if isinstance(x[0], tf.keras.layers.Layer):
            key = lambda layer: layer.name
            return sorted(x, key=key), sorted(y, key=key)
        # Sort elements if they can be transformed to Keras layers
        if(
            hasattr(x[0], 'layer') and
            isinstance(x[0].layer, tf.keras.layers.Layer)
        ):
            p, q = [xi.layer for xi in x], [yi.layer for yi in y]
            return ModelSerializationTest.sort_lists_or_tuples(p, q)
        else:  # Return unsorted lists because no sort strategy was determined
            return x, y


    @staticmethod
    def model_output_validation(original_y, deserial_y):
        """
        Determine whether the outputs are equal or not.

        :param original_y: The output (predictions) of the original model.
        :type original_y: :class:`np.ndarray`
        :param deserial_y: The output (predictions) of the deserialized model.
        :type deserial_y: :class:`np.ndarray`
        :return: True if the outputs are valid (equal), False otherwise.
        :rtype: bool
        """
        # Always passed when no outputs are given
        if original_y is None and deserial_y is None:
            return True
        # When outputs are not None, check for equality
        return np.allclose(original_y, deserial_y)

