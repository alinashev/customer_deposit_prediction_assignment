import lightgbm as lgb

from customer_deposit_prediction_assignment.models.model.base_model import BaseModel


class LGBMModel(BaseModel):
    """LightGBM model wrapper for binary classification.

    This class wraps LightGBM's `LGBMClassifier` and integrates it with the `BaseModel` interface.
    It supports optional parameter passing and optional cross-validation setup.

    Inherits from:
        BaseModel: A custom base class for machine learning models with a unified interface.

    Attributes:
        params (dict): Parameters passed to the LightGBM model.
        model (lgb.LGBMClassifier): The underlying LightGBM classifier.
    """

    def __init__(self, X_train, y_train, X_val, y_val, params=None, enable_cv=False, cv_params=None, **kwargs):
        """Initializes the LGBMModel with training and validation data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.Series or np.ndarray): Validation labels.
            params (dict, optional): Parameters for the LGBMClassifier. Defaults to None.
            enable_cv (bool, optional): Whether to enable cross-validation. Defaults to False.
            cv_params (dict, optional): Parameters for cross-validation if enabled.
            **kwargs: Additional keyword arguments passed to the BaseModel.
        """
        super().__init__(X_train, y_train, X_val, y_val, enable_cv=enable_cv, cv_params=cv_params, **kwargs)
        self.params = params if params else {}
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self):
        """Trains the LightGBM model on the training data.

        Returns:
            LGBMModel: The instance of the trained model (self).
        """
        self.model.fit(self.X_train, self.y_train)
        return self
