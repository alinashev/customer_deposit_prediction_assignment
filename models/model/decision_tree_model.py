from sklearn.tree import DecisionTreeClassifier
from customer_deposit_prediction_assignment.models.model.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    """Decision Tree model wrapper for binary classification.

    This class wraps scikit-learn's `DecisionTreeClassifier` and integrates it with
    the custom `BaseModel` interface. It supports optional cross-validation and
    passes additional keyword arguments to the underlying model.

    Inherits from:
        BaseModel: A custom base class for ML models that provides a consistent API.

    Attributes:
        model (DecisionTreeClassifier): The scikit-learn decision tree model instance.
    """

    def __init__(self, X_train, y_train, X_val, y_val, max_depth=None, criterion="gini",
                 enable_cv=False, cv_params=None, **kwargs):
        """Initializes the DecisionTreeModel with training and validation data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.Series or np.ndarray): Validation labels.
            max_depth (int, optional): Maximum depth of the decision tree. Defaults to None.
            criterion (str, optional): Function to measure the quality of a split.
                Supported values: 'gini' (default), 'entropy', 'log_loss'.
            enable_cv (bool, optional): Whether to perform cross-validation. Defaults to False.
            cv_params (dict, optional): Cross-validation configuration parameters.
            **kwargs: Additional keyword arguments passed to `DecisionTreeClassifier`.
        """
        super().__init__(X_train, y_train, X_val, y_val, enable_cv=enable_cv, cv_params=cv_params, **kwargs)
        self.model = DecisionTreeClassifier(max_depth=max_depth,
                                            criterion=criterion,
                                            **kwargs)

    def fit(self):
        """Trains the decision tree model on the training data.

        If cross-validation is enabled, it is run before fitting the model.

        Returns:
            DecisionTreeModel: The instance of the trained model (self).
        """
        self.cross_validate()
        self.model.fit(self.X_train, self.y_train)
        return self
