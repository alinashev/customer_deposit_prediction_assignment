from sklearn.neighbors import KNeighborsClassifier

from customer_deposit_prediction_assignment.models.model.base_model import BaseModel


class KNNModel(BaseModel):
    """K-Nearest Neighbors (KNN) model wrapper for binary classification.

    This class wraps scikit-learn's `KNeighborsClassifier` and integrates it with
    the custom `BaseModel` interface. It supports optional data transformation using
    an external enricher (e.g., a pipeline or preprocessor).

    Inherits from:
        BaseModel: A custom base class that standardizes model interface.

    Attributes:
        enricher (object or None): Optional transformer with `fit_transform` and `transform` methods.
        model (KNeighborsClassifier): The underlying scikit-learn KNN classifier.
    """

    def __init__(self, X_train, y_train, X_val, y_val, n_neighbors=5, enricher=None, **kwargs):
        """Initializes the KNNModel with training and validation data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.Series or np.ndarray): Validation labels.
            n_neighbors (int, optional): Number of neighbors to use. Defaults to 5.
            enricher (object, optional): Optional transformer object with `.fit_transform()` and `.transform()`.
            **kwargs: Additional keyword arguments passed to `KNeighborsClassifier`.
        """
        self.enricher = enricher
        X_train_transformed = self.enricher.fit_transform(X_train) if self.enricher else X_train
        X_val_transformed = self.enricher.transform(X_val) if self.enricher else X_val

        super().__init__(X_train_transformed, y_train, X_val_transformed, y_val, **kwargs)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)

    def fit(self):
        """Fits the KNN model on the training data.

        Returns:
            KNNModel: The instance of the trained model (self).
        """
        self.model.fit(self.X_train, self.y_train)
        return self
