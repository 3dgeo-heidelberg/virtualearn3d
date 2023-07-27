# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.handle.dl_model_handler import DLModelHandler
from src.report.deep_learning_model_summary_report import \
    DeepLearningModelSummaryReport
from src.utils.dict_utils import DictUtils
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
import tensorflow as tf
import numpy as np
import time


# ---   CLASS   --- #
# ----------------- #
class SimpleDLModelHandler(DLModelHandler):
    """
    # TODO Rethink : Document class and ivars
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, arch, **kwargs):
        # TODO Rethink : doc
        # Call parent's init
        super().__init__(arch, **kwargs)
        # Assign member attributes
        self.summary_report_path = kwargs.get('summary_report_path', None)
        self.out_prefix = kwargs.get('out_prefix', None)
        self.training_epochs = kwargs.get('epochs', 100)
        self.history = kwargs.get('history', None)
        self.checkpoint_path = kwargs.get('checkpoint_path', None)
        self.checkpoint_monitor = kwargs.get('checkpoint_monitor', 'loss')
        self.compilation_args = kwargs.get('compilation_args', {
            'optimizer': {
                'algorithm': 'SGD',
                'learning_rate': 1e-3,
            },
            'loss': {
                'function': 'sparse_categorical_crossentropy'
            },
            'metrics': [
                'sparse_categorical_accuracy'
            ]
        })

    # ---   MODEL HANDLER   --- #
    # ------------------------- #
    def _fit(self, X, y, F=None):
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
        # Fit the model
        start = time.perf_counter()
        self.history = self.compiled.fit(
            *self.arch.run_pre({'X': X, 'y': y}),
            epochs=self.training_epochs,
            callbacks=callbacks,
            batch_size=16  # TODO Rethink : Externalize argument
        )
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Deep learning model trained on {X.shape[0]} points during '
            f'{self.training_epochs} epochs in {end-start:.3f} seconds.'
        )
        # Take best model from checkpoint
        if self.checkpoint_path is not None:
            self.compiled.load_weights(self.checkpoint_path)
        # Plot history
        # TODO Rethink : Implement
        # Return
        return self

    def _predict(self, X, F=None):
        # TODO Rethink : Sphinx doc
        # Softmax scores
        zhat = self.arch.run_post({
            'X': X,
            'z': self.compiled.predict(self.arch.run_pre({'X': X})),
        })
        print(f'zhat: {zhat}')  # TODO Remove
        print(f'zhat.shape: {zhat.shape}')  # TODO Remove
        # Final predictions
        yhat = np.argmax(zhat, axis=1)
        # Return
        return yhat
        # TODO Rethink : Implement

    def compile(self, X=None, y=None, F=None):
        """
        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler.compile`.
        """
        if not self.arch.is_built():
            self.arch.build()
        self.compiled = self.arch.nn
        self.compiled.compile(
            **SimpleDLModelHandler.build_compilation_args(
                self.compilation_args
            )
        )
        return self

    # ---  UTIL METHODS  --- #
    # ---------------------- #
    @staticmethod
    def build_compilation_args(comp_args):
        # TODO Rethink : Sphinx doc
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
        for metric_name in metrics:
            metric_class = None
            if metric_name == 'sparse_categorical_accuracy':
                metric_class = tf.keras.metrics.sparse_categorical_accuracy
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
