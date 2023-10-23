Introduction
**************

The VirtuaLearn3D (VL3D) framework is a software for point-wise classification and regression on 3D point clouds. It handles the training and prediction operations of the many implemented models. On top of that, it provides useful tools for transforming the point clouds and for automatic feature extraction. Moreover, it also supports evaluations on the data, the models, and the predictions. These evaluations yield further insights and analysis through automatically generated plots, CSV files, text reports, and point clouds.

The VL3D framework is designed so the user can take model-selection and architecture-design choices and define custom pipelines. In doing so, it is enough to manipulate JSON files like the one shown below:

.. code-block:: json

   {
      "in_pcloud": [
        "test_data/QuePet_BR01_01_2019-07-03_q2_TLS-on_c.laz",
        "test_data/QueRub_KA11_09_2019-09-03_q2_TLS-on_c_t.laz"
      ],
      "out_pcloud": [
        "out/training/QuePet_BR01_pca_RF/*",
        "out/training/QueRub_KA11_09_pca_RF/*"
      ],
      "sequential_pipeline": [
        {
          "miner": "GeometricFeatures",
          "radius": 0.05,
          "fnames": ["linearity", "planarity", "surface_variation", "eigenentropy", "omnivariance", "verticality", "anisotropy"]
        },
        {
          "miner": "GeometricFeatures",
          "radius": 0.1,
          "fnames": ["linearity", "planarity", "surface_variation", "eigenentropy", "omnivariance", "verticality", "anisotropy"]
        },
        {
          "miner": "GeometricFeatures",
          "radius": 0.2,
          "fnames": ["linearity", "planarity", "surface_variation", "eigenentropy", "omnivariance", "verticality", "anisotropy"]
        },
        {
          "writer": "Writer",
          "out_pcloud": "*pcloud/geomfeats.laz"
        },
        {
          "imputer": "UnivariateImputer",
          "fnames": ["AUTO"],
          "target_val": "NaN",
          "strategy": "mean",
          "constant_val": 0
        },
        {
          "feature_transformer": "Standardizer",
          "fnames": ["AUTO"],
          "center": true,
          "scale": true
        },
        {
          "feature_transformer": "PCATransformer",
          "out_dim": 0.99,
          "whiten": false,
          "random_seed": null,
          "fnames": ["AUTO"],
          "report_path": "*report/pca_projection.log",
          "plot_path": "*plot/pca_projection.svg"
        },
        {
          "writer": "Writer",
          "out_pcloud": "*pcloud/geomfeats_transf.laz"
        },
        {
          "train": "RandomForestClassifier",
          "fnames": ["AUTO"],
          "training_type": "stratified_kfold",
          "random_seed": null,
          "shuffle_points": true,
          "num_folds": 5,
          "model_args": {
            "n_estimators": 4,
            "criterion": "entropy",
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": null,
            "min_impurity_decrease": 0.0,
            "bootstrap": true,
            "oob_score": false,
            "n_jobs": 4,
            "warm_start": false,
            "class_weight": null,
            "ccp_alpha": 0.0,
            "max_samples": 0.8
          },
          "autoval_metrics": ["OA", "P", "R", "F1", "IoU", "wP", "wR", "wF1", "wIoU", "MCC", "Kappa"],
          "stratkfold_report_path": "*report/RF_stratkfold_report.log",
          "stratkfold_plot_path": "*plot/RF_stratkfold_plot.svg",
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
            "report_path": "*report/RF_hyper_grid_search.log"
          },
          "importance_report_path": "*report/LeafWood_Training_RF_importance.log",
          "importance_report_permutation": true,
          "decision_plot_path": "*plot/LeafWood_Training_RF_decission.svg",
          "decision_plot_trees": 3,
          "decision_plot_max_depth": 5
        },
        {
          "writer": "PredictivePipelineWriter",
          "out_pipeline": "*pipe/LeafWood_Training_RF.pipe",
          "include_writer": false,
          "include_imputer": true,
          "include_feature_transformer": true,
          "include_miner": true
        }
      ]
    }

The JSON below defines a pipeline to train random forest models. It will be run twice, once to train the model on the QuePet_BR01 tree and once to train on the QueRub_KA11 tree. Three sets of geometric features are computed with different radii for each input point cloud. The generated features are then written to an output point cloud **geomfeats.laz** to visualize them. The mean value of the feature will replace any feature with an invalid numerical value through the univariate imputer. Afterward, the features are standardized to have mean zero and standard deviation one. Then, the dimensionality of the feature space is transformed through PCA, and the resulting transformed features are exported to **geomfeats_transf.laz** for visualization.

At this point, the features are used to train a random forest classifier using a stratified kfold training strategy with :math:`k=5`. The trained model is evaluated through metrics like Overall Accuracy (OA) or Matthews Correlation Coefficient (MCC). Some model hyperparameters, like the number of estimators or the max depth of each decision tree, are explored using a grid search algorithm. The best combination of hyperparameters is automatically selected to train the final model. Finally, the data mining, imputation, and feature transformation components are assembled with the random forest classifier, and serialized to a file **LeafWood_Training_RF.pipe** that can be later loaded to be used as a leaf-wood segmentation model.

**TODO:** *Add images with results*

**TODO:** *Link to further documentation*
