# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.handle.dl_model_handler import DLModelHandler
from src.report.deep_learning_model_summary_report import \
    DeepLearningModelSummaryReport
from src.report.receptive_fields_report import ReceptiveFieldsReport
from src.report.training_history_report import TrainingHistoryReport
from src.plot.training_history_plot import TrainingHistoryPlot
from src.utils.dict_utils import DictUtils
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
import time


# ---   CLASS   --- #
# ----------------- #
class SimpleDLModelHandler(DLModelHandler):
    """
    Class to handle deep learning models in a simple way. It can be seen as the
    baseline deep learning model handler. See :class:`.DLModelHandler`.

    :ivar summary_report_path: Path to the file where the summary report
        will be written, i.e., the report that summarizes the model's
        architecture.
    :vartype summary_report_path: str
    :ivar training_history_dir: Path to the directory where the training
        history plots and reports will be exported, i.e., information related
        to the  training along many epochs.
    :vartype training_history_dir: path
    :ivar out_prefix: The output prefix for path expansions, when necessary.
    :vartype out_prefix: str
    :ivar training_epochs: The number of training epochs for fitting the model.
    :vartype training_epochs: int
    :ivar batch_size: The batch size governing the model's input.
    :vartype batch_size: int
    :ivar history: By default None. It will be updated to contain the training
        history when calling fit.
    :vartype history: None or :class:`tf.keras.callbacks.History`
    :ivar checkpoint_path: The path where the model's checkpoint will be
        exported. It is used to keep the best model when using the checkpoint
        callback strategy during training.
    :vartype checkpoint_path: str
    :ivar checkpoint_monitor: The name of the metric to choose the best
        model. By default, it is "loss", which represents the loss function.
    :vartype checkpoint_monitor: str
    :ivar learning_rate_on_plateau: The key-word arguments governing the
        instantiation of the learning rate on plateau callback.
    :vartype learning_rate_on_plateau: dict
    :ivar early_stopping: The key-word arguments governing the instantiation
        of the early stopping callback.
    :vartype early_stopping: dict
    :ivar compilation_args: The specification on how to compile the model.
        See
        :meth:`simple_dl_model_handler.SimpleDLModelHandler.build_compilation_args`
        .
    :vartype compilation_args: dict
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, arch, **kwargs):
        """
        Initialize/instantiate a simple deep learning model handler.

        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler.__init__`.
        """
        # Call parent's init
        super().__init__(arch, **kwargs)
        # Assign member attributes
        self.summary_report_path = kwargs.get('summary_report_path', None)
        self.training_history_dir = kwargs.get('training_history_dir', None)
        self.out_prefix = kwargs.get('out_prefix', None)
        self.training_epochs = kwargs.get('training_epochs', 100)
        self.batch_size = kwargs.get('batch_size', 16)
        self.history = kwargs.get('history', None)
        self.checkpoint_path = kwargs.get('checkpoint_path', None)
        self.checkpoint_monitor = kwargs.get('checkpoint_monitor', 'loss')
        self.learning_rate_on_plateau = kwargs.get(
            'learning_rate_on_plateau',
            None
        )
        self.early_stopping = kwargs.get('early_stopping', None)
        self.compilation_args = kwargs.get('compilation_args', None)

    # ---   MODEL HANDLER   --- #
    # ------------------------- #
    def _fit(self, X, y, F=None):
        """
        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler._fit`.
        """
        # Report the model
        summary = DeepLearningModelSummaryReport(self.compiled)
        LOGGING.LOGGER.info(summary.to_string())
        if self.summary_report_path is not None:
            summary.to_file(
                self.summary_report_path,
                out_prefix=self.out_prefix
            )
            LOGGING.LOGGER.info(
                'Deep learning model summary written to "'
                f'{self.summary_report_path}"'
            )
        # Handle training callbacks
        callbacks = []
        if self.checkpoint_path is not None:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                self.checkpoint_path,
                monitor=self.checkpoint_monitor,
                save_best_only=True,
                save_weights_only=True
            ))
        if self.learning_rate_on_plateau is not None:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                **self.learning_rate_on_plateau
            ))
        if self.early_stopping is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                **self.early_stopping
            ))
        # Fit the model
        start = time.perf_counter()
        X, y = self.arch.run_pre({'X': X, 'y': y})
        class_weight = self.handle_class_weight(y)
        y = self.handle_labels_format(y)
        self.history = self.compiled.fit(
            X, y,
            epochs=self.training_epochs,
            callbacks=callbacks,
            batch_size=self.batch_size,
            class_weight=class_weight
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Deep learning model trained on {X.shape[0]} points during '
            f'{self.training_epochs} epochs in {end-start:.3f} seconds.'
        )
        # Take best model from checkpoint
        if self.checkpoint_path is not None:
            self.compiled.load_weights(self.checkpoint_path)
        # Report and plot history
        if self.training_history_dir is not None:
            report_path = os.path.join(
                self.training_history_dir, 'training_history.csv'
            )
            TrainingHistoryReport(
                self.history
            ).to_file(report_path, out_prefix=self.out_prefix)
            LOGGING.LOGGER.info(
                'Deep learning training history report written to '
                f'"{report_path}"'
            )
            TrainingHistoryPlot(
                self.history,
                path=self.training_history_dir
            ).plot(
                out_prefix=self.out_prefix
            )
            LOGGING.LOGGER.info(
                'Deep learning training history plots exported to '
                f'"{self.training_history_dir}"'
            )
        # Return
        return self

    def _predict(self, X, F=None, y=None, zout=None):
        """
        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler._predict`.
        """
        # Softmax scores
        X_rf = self.arch.run_pre({'X': X})
        zhat_rf = self.compiled.predict(X_rf, batch_size=self.batch_size)
        zhat = self.arch.run_post({'X': X, 'z': zhat_rf})
        if zout is not None:  # When z is not None it must be a list
            zout.append(zhat)  # Append propagated zhat to z list
        # Final predictions
        yhat = np.argmax(zhat, axis=1) if len(zhat.shape) > 1 \
            else np.round(zhat)
        # Report receptive fields, if requested
        rf_dir = getattr(
            self.arch.pre_runnable.pre_processor,
            'receptive_fields_dir',
            None
        ) if getattr(
            self.arch.pre_runnable, 'pre_processor', None
        ) is not None else None
        if rf_dir is not None:
            ReceptiveFieldsReport(
                X_rf=X_rf,  # X (for each receptive field)
                zhat_rf=zhat_rf,  # Softmax scores (for each receptive field)
                yhat_rf=np.array([  # Predictions (for each receptive field)
                    np.argmax(zhat_rf_i, axis=1)
                    for zhat_rf_i in zhat_rf
                ]),
                y_rf=self.arch.pre_runnable.pre_processor.reduce_labels(
                    # Reduced expected classes (for each receptive field)
                    X_rf, y
                ) if y is not None else None,
                class_names=self.class_names
            ).to_file(rf_dir, self.out_prefix)
        # Return
        return yhat

    def compile(self, X=None, y=None, F=None):
        """
        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler.compile`.
        """
        if not self.arch.is_built():
            self.arch.build()
        self.compiled = self.arch.nn
        self.compiled.compile(
            run_eagerly=True,  # Uncomment for better debugging (but slower)
            **SimpleDLModelHandler.build_compilation_args(
                self.compilation_args
            )
        )
        return self

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    @staticmethod
    def build_compilation_args(comp_args):
        """
        Build the compilation arguments from given spec.

        :param comp_args: The specification to build the compilation arguments.
        :return: The dictionary of compilation arguments.
        :rtype: dict
        """
        # Build optimizer : Extract args
        opt_args = comp_args['optimizer']
        opt_alg = opt_args['algorithm'].lower()
        opt_lr = opt_args.get('learning_rate', None)
        # Build optimizer : Determine class (algorithm)
        optimizer = None
        if opt_alg == 'sgd':
            optimizer = tf.keras.optimizers.SGD
        if optimizer is None:
            raise DeepLearningException(
                'SimpleDLModelHandler cannot compile a model without an '
                'optimizer. None was given.'
            )
        # Build optimizer
        optimizer = optimizer(**DictUtils.delete_by_val({
            'learning_rate': opt_lr
        }, None))
        # Build loss : Extract args
        loss_args = comp_args['loss']
        loss_fun = loss_args['function'].lower()
        # Build loss : Determine class (function)
        loss = None
        if loss_fun == 'sparse_categorical_crossentropy':
            loss = tf.keras.losses.SparseCategoricalCrossentropy
        if loss_fun == 'binary_crossentropy':
            loss = tf.keras.losses.BinaryCrossentropy
        if loss_fun == 'categorical_crossentropy':
            loss = tf.keras.losses.CategoricalCrossentropy
        if loss is None:
            raise DeepLearningException(
                'SimpleDLModelHandler cannot compile a model without a loss '
                'function. None was given.'
            )
        # Build loss
        loss = loss()
        # Build metrics : Extract args
        metrics_args = comp_args['metrics']
        # Build metrics : Determine metrics (list of classes)
        metrics = []
        for metric_name in metrics_args:
            metric_class = None
            if metric_name == 'sparse_categorical_accuracy':
                metric_class = tf.keras.metrics.sparse_categorical_accuracy
            if metric_name == 'binary_accuracy':
                metric_class = tf.keras.metrics.binary_accuracy
            if metric_name == 'precision':
                metric_class = tf.keras.metrics.Precision(name='precision')
            if metric_name == 'recall':
                metric_class = tf.keras.metrics.Recall(name='recall')
            if metric_class is None:
                raise DeepLearningException(
                    'SimpleDLModelHandler cannot compile a model because a '
                    f'given metric cannot be interpreted ("{metric_name}").'
                )
            metrics.append(metric_class)
        if len(metrics) < 1:
            LOGGING.LOGGER.debug(
                'SimpleDLModelHandler detected a model compilation with no '
                'metrics. While this is supported, recall an arbitrary number '
                'of evaluation metrics can be used to evaluate the training '
                'performance. These metrics can be more easy to interpret or '
                'bring further insights into the model than the loss function '
                'alone.'
            )
        # Return dictionary of built compilation args
        return {
            'optimizer': optimizer,
            'loss': loss,
            'metrics': metrics
        }

    def handle_class_weight(self, y):
        """
        # TODO Rethink : Doc
        """
        # No class weight specification
        if self.class_weight is None:
            return None
        # Handle class weight specification
        if self.class_weight == "auto":  # Automatic
            num_classes = getattr(self.arch, "num_classes", None)
            if num_classes is None:
                raise DeepLearningException(
                    'SimpleDLModelHandler does not support automatic class '
                    'weight for current architecture: '
                    f'"{self.arch.__class__.__name__}"'
                )
            keys = [class_id for class_id in range(num_classes)]
            num_samples = len(y)
            num_samples_per_class = np.array([
                np.count_nonzero(y == class_id) for class_id in keys
            ], dtype=int)
            vals = num_samples/num_samples_per_class/num_classes
            return dict(zip(keys, vals))
        else:  # User-given
            return self.class_weight

    def handle_labels_format(self, y):
        """
        # TODO Rethink : Doc
        """
        # Extract loss function name
        loss_low = self.compilation_args['loss']['function'].lower()
        # Handle loss functions that demand one-hot labels
        if (
            loss_low == 'categorical_crossentropy' or
            loss_low == 'binary_crossentropy'
        ):  # Handle one hot encoding for labels
            num_classes = getattr(self.arch, "num_classes", None)
            if num_classes is None:
                raise DeepLearningException(
                    'SimpleDLModelHandler does not support categorical or '
                    'binary crossentropy without a priori specifying the '
                    'number of classes.'
                )
            label_binarizer = LabelBinarizer().fit([
                class_id for class_id in range(num_classes)
            ])
            new_y = []
            for i in range(len(y)):
                new_y.append(label_binarizer.transform(y[i].flatten()))
            y = np.array(new_y)
        if (
            loss_low == 'sparse_categorical_crossentropy' and
            self.class_weight is not None
        ):
            raise DeepLearningException(
                'SimpleDLModelHandler detected that class weight is requested '
                'for a sparse categorical crossentropy loss. Currently, this '
                'is not supported.'
            )
        # By default, labels can be used straight forward
        return y
