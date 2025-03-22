class HyperparameterTuner:
    """Base class for hyperparameter tuning of machine learning models.

    This class provides a common interface for tuning hyperparameters on a training
    and validation dataset. Specific tuning strategies should be implemented
    in child classes by overriding the `tune` method.

    Attributes:
        model_class (type): The class of the model to be tuned (e.g., LGBMClassifier).
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        X_val (pd.DataFrame or np.ndarray): Validation features.
        y_val (pd.Series or np.ndarray): Validation labels.
        param_space (dict): The hyperparameter search space.
        max_evals (int): Maximum number of optimization evaluations.
        best_params (dict or None): Best parameters found after tuning.
    """

    def __init__(self, model_class, X_train, y_train, X_val, y_val, param_space, max_evals=20):
        """Initializes the HyperparameterTuner.

        Args:
            model_class (type): The class of the model to tune.
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training target values.
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.Series or np.ndarray): Validation target values.
            param_space (dict): The hyperparameter search space.
            max_evals (int, optional): Maximum number of optimization evaluations. Defaults to 20.
        """
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.param_space = param_space
        self.max_evals = max_evals
        self.best_params = None

    def tune(self):
        """Runs the hyperparameter tuning process.

        This method must be implemented in a subclass.

        Raises:
            NotImplementedError: Always, to enforce implementation in a subclass.
        """
        raise NotImplementedError("The tune() method must be implemented in a child class.")
