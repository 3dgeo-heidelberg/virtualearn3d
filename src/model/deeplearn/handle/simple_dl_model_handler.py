# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.handle.dl_model_handler import DLModelHandler
from src.model.deeplearn.loss.class_weighted_binary_crossentropy import \
    vl3d_class_weighted_binary_crossentropy
from src.model.deeplearn.loss.class_weighted_categorical_crossentropy import \
    vl3d_class_weighted_categorical_crossentropy
from src.report.deep_learning_model_summary_report import \
    DeepLearningModelSummaryReport
from src.report.receptive_fields_report import ReceptiveFieldsReport
from src.report.receptive_fields_distribution_report import \
    ReceptiveFieldsDistributionReport
from src.plot.receptive_fields_distribution_plot import \
    ReceptiveFieldsDistributionPlot
from src.report.training_history_report import TrainingHistoryReport
from src.plot.training_history_plot import TrainingHistoryPlot
from src.utils.dict_utils import DictUtils
from src.inout.io_utils import IOUtils
from src.model.deeplearn.deep_learning_exception import DeepLearningException
import src.main.main_logger as LOGGING
import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError as \
    TFResourceExhaustedError
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import copy
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
    :ivar fit_verbose: Whether to use silent mode (0), show a progress bar (1),
        or print one line per epoch (2). Alternatively, "auto" can be used
        which typically means (1).
    :vartype fit_verbose: str or int
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
        # TODO Rethink : RBF and FSL dirs should belong to the model? externalize them somehow?
        self.feat_struct_repr_dir = kwargs.get(
            'features_structuring_representation_dir', None
        )
        self.rbf_feat_extract_repr_dir = kwargs.get(
            'rbf_feature_extraction_representation_dir', None
        )
        self.rbf_feat_processing_repr_dir = kwargs.get(
            'rbf_feature_processing_representation_dir', None
        )
        self.kpconv_representation_dir = kwargs.get(
            'kpconv_representation_dir', None
        )
        self.skpconv_representation_dir = kwargs.get(
            'skpconv_representation_dir', None
        )
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
        self.fit_verbose = kwargs.get('fit_verbose', "auto")
        self.predict_verbose = kwargs.get('predict_verbose', "auto")

    # ---   MODEL HANDLER   --- #
    # ------------------------- #
    def _fit(self, X, y):
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
        callbacks = self.build_callbacks()
        # Fit the model
        start = time.perf_counter()
        X, y_rf = self.arch.run_pre({
            'X': X,
            'y': y,
            'training_support_points': True
        })
        self.compile(y_rf=y_rf)  # Recompile with y_rf for class weights
        y_rf = self.handle_labels_format(y_rf)  # Label format depends on loss
        self.fit_logic(X, y_rf, callbacks)
        end = time.perf_counter()
        LOGGING.LOGGER.info(
            f'Deep learning model trained on {len(y_rf)} cases during '
            f'{self.training_epochs} epochs in {end-start:.3f} seconds.'
        )
        # Take best model from checkpoint
        if self.checkpoint_path is not None:
            try:
                IOUtils.validate_path_to_file(
                    self.checkpoint_path,
                    msg='Cannot find DL model checkpoint at:'
                )
                self.compiled.load_weights(self.checkpoint_path)
            except FileNotFoundError as fnferr:
                LOGGING.LOGGER.warning(
                    'SimpleDLModelHandler failed to restore DL model weights '
                    f'from "{self.checkpoint_path}".'
                )
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
        # Predictions on the training receptive fields for plots and reports
        if getattr(self.arch.pre_runnable, 'pre_processor', None) is not None:
            if getattr(
                self.arch.pre_runnable.pre_processor,
                'training_receptive_fields_dir',
                None
            ) is not None or getattr(
                self.arch.pre_runnable.pre_processor,
                'training_receptive_fields_distribution_report_path',
                None
            ) is not None or getattr(
                self.arch.pre_runnable.pre_processor,
                'training_receptive_fields_distribution_plot_path',
                None
            ):
                zhat = self.compiled.predict(X, batch_size=self.batch_size)
                X_rf = X[0] if isinstance(X, list) else X
                self.handle_receptive_fields_plots_and_reports(
                    X_rf=X_rf,
                    zhat_rf=zhat,
                    y=y,
                    training=True
                )
        # Return
        return self

    def fit_logic(self, X, y_rf, callbacks):
        """
        Implement the fit logic.

        :param X: The pre-processed input.
        :param y_rf: The point-wise labels for each receptive field.
        :param callbacks: The callbacks to be executed during fit.
        :return: The fit history (also assigned internally to self.history).
        """
        # Map for caching stuff during fit logic
        fit_cache_map = {
            'fsl_dir_path': self.feat_struct_repr_dir,
            'rbf_dir_path': self.rbf_feat_extract_repr_dir,
            'rbf_feat_processing_dir_path': self.rbf_feat_processing_repr_dir,
            'kpconv_representation_dir': self.kpconv_representation_dir,
            'skpconv_representation_dir': self.skpconv_representation_dir,
            'out_prefix': self.out_prefix,
            'X': X,
            'y_rf': y_rf,
            'training_epochs': self.training_epochs,
            'callbacks': callbacks,
            'batch_size': self.batch_size,
            'compilef': lambda _y_rf: self.compile(y_rf=_y_rf)
        }
        # Pre-fit logic
        if hasattr(self.arch, 'prefit_logic_callback'):
            self.arch.prefit_logic_callback(fit_cache_map)
        # Fit logic
        self.history = self.compiled.fit(
            X, y_rf,
            epochs=self.training_epochs,
            callbacks=callbacks,
            batch_size=self.batch_size,
            verbose=self.fit_verbose
        )
        # Post-fit logic
        if hasattr(self.arch, 'posfit_logic_callback'):
            # Update callbacks for freeze training iterations
            callbacks = self.build_callbacks()
            fit_cache_map['callbacks'] = callbacks
            fit_cache_map['history'] = self.history
            self.arch.posfit_logic_callback(fit_cache_map)
            self.history = fit_cache_map['history']
        return self.history

    def _predict(self, X, y=None, zout=None, plots_and_reports=True):
        """
        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler._predict`.
        """
        # Softmax scores
        X_rf = self.arch.run_pre({
            'X': X,
            'support_points': True,
            'plots_and_reports': plots_and_reports
        })
        try:
             zhat_rf = self.compiled.predict(
                X_rf,
                batch_size=self.batch_size,
                verbose=self.predict_verbose
            )
        except TFResourceExhaustedError as resexherr:
            LOGGING.LOGGER.debug(
                'SimpleDLModelHandler could not compute predictions for '
                f'{X_rf.shape} points using the GPU.\n'
                'Trying CPU instead ...'
            )
            with tf.device("cpu:0"):
                zhat_rf = self.compiled.predict(
                    X_rf, batch_size=self.batch_size
                )
        zhat = self.arch.run_post({'X': X, 'z': zhat_rf})
        if zout is not None:  # When z is not None it must be a list
            zout.append(zhat)  # Append propagated zhat to z list

        # Final predictions
        yhat = np.argmax(zhat, axis=1) if len(zhat.shape) > 1 \
            else np.round(zhat)
        # Do plots and reports
        if plots_and_reports:
            _X_rf = X_rf[0] if isinstance(X_rf, list) else X_rf
            self.handle_receptive_fields_plots_and_reports(
                X_rf=_X_rf,
                zhat_rf=zhat_rf,
                y=y
            )
        # Return
        return yhat

    def compile(self, X=None, y=None, y_rf=None, **kwargs):
        """
        See :class:`.DLModelHandler` and
        :meth:`dl_model_handler.DLModelHandler.compile`.

        :param y_rf: The expected values for each receptive field. Can be
            used to derive class weights.
        """
        # Build architecture
        if not self.arch.is_built():
            self.arch.build()
        elif kwargs.get('arch_plot', False):  # If not,
            # at least plot the built architecture if requested
            self.arch.plot()
        self.compiled = self.arch.nn
        # Determine class weights if possible
        class_weight = None
        if y_rf is not None:
            class_weight = self.handle_class_weight(y_rf)
        # Build compilation args
        comp_args = SimpleDLModelHandler.build_compilation_args(
            self.compilation_args
        )
        if class_weight is not None:  # Recompile for custom class weight loss
            comp_args['loss'] = comp_args['loss'](
                np.array(list(class_weight.values()), dtype=np.float32)
            )
        # Compile
        self.compiled.compile(
            run_eagerly=True,  # Uncomment for better debugging (but slower)
            **comp_args
        )
        return self

    def overwrite_pretrained_model(self, spec):
        """
        See :meth:`dl_model_handler.DLModelHandler.overwrite_pretrained_model`.
        """
        # Call parent's method
        super().overwrite_pretrained_model(spec)
        # Overwrite attributes of simple deep learning model handler
        spec_keys = spec.keys()
        if 'model_handling' in spec_keys:
            spec_handling = spec['model_handling']
            spec_handling_keys = spec_handling.keys()
            if 'summary_report_path' in spec_handling_keys:
                self.summary_report_path = spec_handling['summary_report_path']
            if 'training_history_dir' in spec_handling_keys:
                self.training_history_dir = spec_handling['training_history_dir']
            if 'checkpoint_path' in spec_handling_keys:
                self.checkpoint_path = spec_handling['checkpoint_path']
            if 'checkpoint_monitor' in spec_handling_keys:
                self.checkpoint_monitor = spec_handling['checkpoint_monitor']
            if 'batch_size' in spec_handling_keys:
                self.batch_size = spec_handling['batch_size']
            if 'training_epochs' in spec_handling_keys:
                self.training_epochs = spec_handling['training_epochs']
            if 'learning_rate_on_plateau' in spec_handling_keys:
                self.learning_rate_on_plateau = \
                    spec_handling['learning_rate_on_plateau']
            if 'early_stopping' in spec_handling_keys:
                self.early_stopping = spec_handling['early_stopping']

    def update_paths(self, model_args):
        """
        Consider the current specification of model handling arguments to
        update the paths.
        """
        # Nothing to do if no specification is given
        if model_args is None:
            return
        # Update model paths
        model_handling = model_args.get('model_handling', None)
        if model_handling is not None:
            self.summary_report_path = model_handling['summary_report_path']
            self.training_history_dir = model_handling['training_history_dir']
            self.checkpoint_path = model_handling['checkpoint_path']
            self.feat_struct_repr_dir = model_handling.get(
                'features_structuring_representation_dir',
                self.feat_struct_repr_dir
            )
            self.rbf_feat_extract_repr_dir = model_handling.get(
                'rbf_feat_extract_repr_dir',
                self.rbf_feat_extract_repr_dir
            )
            self.rbf_feat_processing_repr_dir = model_handling.get(
                'rbf_feat_processing_repr_dir',
                self.rbf_feat_processing_repr_dir
            )
            self.kpconv_representation_dir = model_handling.get(
                'kpconv_representation_dir', self.kpconv_representation_dir
            )
            self.skpconv_representation_dir = model_handling.get(
                'skpconv_representation_dir', self.skpconv_representation_dir
            )
        # Update architecture paths
        if self.arch is not None:
            self.arch.architecture_graph_path = \
                model_args['architecture_graph_path']
            # Update pre-procesor paths
            pre_processor = None
            if self.arch.pre_runnable is not None:
                if hasattr(self.arch.pre_runnable, "pre_processor"):
                    pre_processor = \
                        self.arch.pre_runnable.pre_processor
            if pre_processor is not None:
                pre_processor.update_paths(model_args.get(
                    'pre_processing', None
                ))

    # ---  MODEL HANDLING TASKS  --- #
    # ------------------------------ #
    def build_callbacks(self):
        """
        See :meth:`dl_model_handler.DLModelHandler.build_callbacks`.
        """
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
        return callbacks


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
        if opt_alg == 'adam':
            optimizer = tf.keras.optimizers.Adam
        if optimizer is None:
            raise DeepLearningException(
                'SimpleDLModelHandler cannot compile a model without an '
                'optimizer. None was given.'
            )
        # Build optimizer : Handle learning rate
        if isinstance(opt_lr, dict):  # Learning schedule
            lr_sched_type = opt_lr['schedule']
            if lr_sched_type == 'exponential_decay':
                opt_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                    **opt_lr['schedule_args']
                )
            else:
                raise DeepLearningException(
                    'SimpleDLModelHandler received an unexpected learning '
                    f'rate schedule: "{lr_sched_type}".'
                )
        # Build optimizer
        optimizer = optimizer(**DictUtils.delete_by_val({
            'learning_rate': opt_lr,
        }, None))
        # Build loss : Extract args
        loss_args = comp_args['loss']
        loss_fun = loss_args['function'].lower()
        # Build loss : Determine class (function)
        instantiate_loss = True
        loss = None
        if loss_fun == 'sparse_categorical_crossentropy':
            loss = tf.keras.losses.SparseCategoricalCrossentropy
        if loss_fun == 'binary_crossentropy':
            loss = tf.keras.losses.BinaryCrossentropy
        if loss_fun == 'class_weighted_binary_crossentropy':
            loss = vl3d_class_weighted_binary_crossentropy
            instantiate_loss = False  # Instantiate later with class weights
        if loss_fun == 'categorical_crossentropy':
            loss = tf.keras.losses.CategoricalCrossentropy
        if loss_fun == 'class_weighted_categorical_crossentropy':
            loss = vl3d_class_weighted_categorical_crossentropy
            instantiate_loss = False  # Instantiate later with class weights
        if loss is None:
            raise DeepLearningException(
                'SimpleDLModelHandler cannot compile a model without a loss '
                'function. None was given.'
            )
        # Build loss
        if instantiate_loss:
            loss = loss()
        # Build metrics : Extract args
        metrics_args = comp_args['metrics']
        # Build metrics : Determine metrics (list of classes)
        metrics = []
        for metric_name in metrics_args:
            metric_class = None
            if metric_name == 'sparse_categorical_accuracy':
                metric_class = tf.keras.metrics.sparse_categorical_accuracy
            if metric_name == 'categorical_accuracy':
                metric_class = tf.keras.metrics.categorical_accuracy
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
        r"""
        Handle the class weight parameter.

        If no class weight is requested, then class weight will be None.

        If automatic class weight is requested (i.e., "auto"), then the
        class weight is automatically determined from the distribution of
        expected classes to give a greater weight to less frequent classes
        and a smaller weight to more frequent classes. More concretely, let
        :math:`m` be the number of samples, :math:`m_i` be the number of
        samples corresponding to class :math:`i`, and :math:`n` be the number
        of classes. Thus, each class weight will be :math:`w_i = m/(n m_i)`.

        If class weight is a list, tuple or array of weights it will be
        translated to a dictionary such that the first element is the weight
        for the first class, and so on.


        :param y: The vector of expected point-wise classes.
        :type y: :class:`np.ndarray`
        :return: Class weight prepared for the model.
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
            num_samples = np.prod(y.shape)
            num_samples_per_class = np.array([
                np.count_nonzero(y == class_id) for class_id in keys
            ], dtype=int)
            vals = num_samples/num_samples_per_class/num_classes
            class_weight_dict = dict(zip(keys, vals))
            LOGGING.LOGGER.debug(
                'Simple DL model handler automatically generated the '
                f'following dictionary of class weights:\n{class_weight_dict}'
            )
            return class_weight_dict
        else:  # User-given
            return dict(zip(  # List to dict with serial int key
                np.arange(len(self.class_weight), dtype=int),
                self.class_weight
            ))

    def handle_labels_format(self, y):
        """
        Handles the format in which labels must be given to the model.

        For instance, if categorical cross entropy is used, labels must be
        given using one-hot-encoding. However, if sparse categorical cross
        entropy is used, labels must be given as an integer.

        :return: The labels prepared for the model.
        """
        # Extract loss function name
        loss_low = self.compilation_args['loss']['function'].lower()
        # Handle loss functions that demand one-hot labels
        if (
            loss_low == 'categorical_crossentropy' or
            loss_low == 'class_weighted_categorical_crossentropy'
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
        if(
            loss_low == 'binary_crossentropy' and
            self.class_weight is not None
        ):
            raise DeepLearningException(
                'SimpleDLModelHandler detected that class weight is requested '
                'for a binary crossentropy loss. This is not supported. '
                'Please, use "class_weighted_binary_crossentropy" loss '
                'function instead.'
            )
        # By default, labels can be used straight forward
        return y

    def handle_receptive_fields_plots_and_reports(
        self, X_rf, zhat_rf, y=None, training=False
    ):
        """
        Handle any plot and reports related to the receptive fields.

        :param X_rf: The receptive fields such that X_rf[i] is the matrix
            of coordinates representing the points in the i-th receptive field.
        :type X_rf: :class:`np.ndarray`
        :param zhat_rf: The output from the neural network for each receptive
            field.
        :type zhat_rf: :class:`np.ndarray`
        :param y: The expected class for each point (considering original
            points, i.e., not the receptive fields).
        :type y: :class:`np.ndarray`
        :param training: Whether the considered receptive fields are those
            used for training (True) or not (False).
        :type training: bool
        :return: Nothing at all but the plots and reports are exported to
            the corresponding files.
        """
        # Extract output paths (either pointing to files or directories)
        rf_dir, rf_dist_report_path, rf_dist_plot_path = None, None, None
        if getattr(self.arch.pre_runnable, 'pre_processor', None) is not None:
            rf_dir = getattr(
                self.arch.pre_runnable.pre_processor,
                'receptive_fields_dir',
                None
            ) if not training else getattr(
                self.arch.pre_runnable.pre_processor,
                'training_receptive_fields_dir',
                None
            )
            rf_dist_report_path = getattr(
                self.arch.pre_runnable.pre_processor,
                'receptive_fields_distribution_report_path',
                None
            ) if not training else getattr(
                self.arch.pre_runnable.pre_processor,
                'training_receptive_fields_distribution_report_path',
                None
            )
            rf_dist_plot_path = getattr(
                self.arch.pre_runnable.pre_processor,
                'receptive_fields_distribution_plot_path',
                None
            ) if not training else getattr(
                self.arch.pre_runnable.pre_processor,
                'training_receptive_fields_distribution_plot_path',
                None
            )
        # Check at least one plot or report is requested
        if (
            rf_dir is None and
            rf_dist_report_path is None and
            rf_dist_plot_path is None
        ):
            return
        # Compute the predicted and expected classes for each receptive field
        yhat_rf = np.array([  # Predictions (for each receptive field)
            np.argmax(zhat_rf_i, axis=1)
            if len(zhat_rf_i.shape) > 1 and zhat_rf_i.shape[-1] != 1
            else np.round(np.squeeze(zhat_rf_i))
            for zhat_rf_i in zhat_rf
        ])
        y_rf = self.arch.pre_runnable.pre_processor.reduce_labels(
            # Reduced expected classes (for each receptive field)
            X_rf, y
        ) if y is not None else None
        # Report receptive fields, if requested
        if rf_dir is not None:
            ReceptiveFieldsReport(
                X_rf=X_rf,  # X (for each receptive field)
                zhat_rf=zhat_rf,  # Softmax scores (for each receptive field)
                yhat_rf=yhat_rf,  # Predictions (for each receptive field)
                y_rf=y_rf,  # Expected (for each receptive field, can be None)
                class_names=self.class_names
            ).to_file(rf_dir, self.out_prefix)
        # Report receptive fields distribution, if requested
        if rf_dist_report_path:
            ReceptiveFieldsDistributionReport(
                yhat_rf=yhat_rf,  # Predictions (for each receptive field)
                y_rf=y_rf,  # Expected (for each receptive field, can be None)
                class_names=self.class_names
            ).to_file(rf_dist_report_path, self.out_prefix)
        # Plot receptive fields distribution, if requested
        if rf_dist_plot_path:
            ReceptiveFieldsDistributionPlot(
                yhat_rf=yhat_rf,  # Predictions (for each receptive field)
                y_rf=y_rf,  # Expected (for each receptive field, can be None)
                class_names=self.class_names,
                path=rf_dist_plot_path
            ).plot(out_prefix=self.out_prefix, logging=True)

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized simple deep learning
        model handler.

        :return: The state's dictionary of the object
        :rtype: dict
        """
        # Obtain from parent
        state = super().__getstate__()
        # Update
        state['summary_report_path'] = self.summary_report_path
        state['training_history_dir'] = self.training_history_dir
        state['feat_struct_repr_dir'] = self.feat_struct_repr_dir
        state['rbf_feat_extract_repr_dir'] = self.rbf_feat_extract_repr_dir
        state['rbf_feat_processing_repr_dir'] = \
            self.rbf_feat_processing_repr_dir
        state['kpconv_representation_dir'] = self.kpconv_representation_dir
        state['skpconv_representation_dir'] = self.skpconv_representation_dir
        state['out_prefix'] = self.out_prefix
        state['training_epochs'] = self.training_epochs
        state['batch_size'] = self.batch_size
        state['history'] = copy.copy(self.history)
        state['history'].model = None  # Do not serialize keras/tf model
        state['checkpoint_path'] = self.checkpoint_path
        state['checkpoint_monitor'] = self.checkpoint_monitor
        state['learning_rate_on_plateau'] = self.learning_rate_on_plateau
        state['early_stopping'] = self.early_stopping
        state['compilation_args'] = self.compilation_args
        state['fit_verbose'] = self.fit_verbose
        state['predict_verbose'] = self.predict_verbose
        # Return Simple DL Model Handler state (for serialization)
        return state

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized simple deep learning model handler.

        :param state: The state's dictionary of the saved simple deep learning
            model handler.
        :return: Nothing, but modifies the internal state of the object.
        """
        # Call parent
        super().__setstate__(state)
        # Assign member attributes from state dictionary
        self.summary_report_path = state['summary_report_path']
        self.training_history_dir = state['training_history_dir']
        self.feat_struct_repr_dir = state.get('feat_struct_repr_dir', None)
        self.rbf_feat_extract_repr_dir = state.get(
            'rbf_feat_extract_repr_dir', None
        )
        self.rbf_feat_processing_repr_dir = state.get(
            'rbf_feat_processing_repr_dir', None
        )
        self.kpconv_representation_dir = state.get(
            'kpconv_representation_dir', None
        )
        self.skpconv_representation_dir = state.get(
            'skpconv_representation_dir', None
        )
        self.out_prefix = state['out_prefix']
        self.training_epochs = state['training_epochs']
        self.batch_size = state['batch_size']
        self.history = state['history']
        self.checkpoint_path = state['checkpoint_path']
        self.checkpoint_monitor = state['checkpoint_monitor']
        self.learning_rate_on_plateau = state['learning_rate_on_plateau']
        self.early_stopping = state['early_stopping']
        self.compilation_args = state['compilation_args']
        self.fit_verbose = state['fit_verbose']
        self.predict_verbose = state['predict_verbose']
