"""
Module for building, evaluating, and visualizing machine learning models.

This module provides a base class `BaseModel` that includes methods for model training,
evaluation, and visualization of performance metrics.

Dependencies:
    - matplotlib.pyplot
    - models.utils.cross_validator.CrossValidator
    - models.utils.model_metrics.ModelEvaluation, Metric
    - models.utils.visualization (display_confusion_matrix, plot_residuals, plot_roc_auc, plot_pr_auc)
"""

from matplotlib import pyplot as plt
from models.utils.cross_validator import CrossValidator
from models.utils.model_metrics import ModelEvaluation, Metric
from models.utils.visualization import display_confusion_matrix, plot_residuals, plot_roc_auc, plot_pr_auc

class BaseModel:
    """
    A base class for building, evaluating, and visualizing machine learning models.

    Attributes:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        X_val (array-like): Validation data features.
        y_val (array-like): Validation data labels.
        pos_label (int): The positive class label for binary classification (default is 1).
        threshold (float): The threshold used for classifying predictions (default is 0.5).
        model (object): The model instance (should be defined in child classes).
        model_metrics (ModelEvaluation): An instance used for calculating metrics.
        enable_cv (bool): Whether cross-validation is enabled.
        cv_params (dict): Parameters for cross-validation.
        cv_results (dict or None): Stores cross-validation results if enabled.
    """

    def __init__(self, X_train, y_train, X_val, y_val, pos_label: int = 1, threshold: float = 0.5,
                 enricher=None, enable_cv: bool = False, cv_params: dict = None, **kwargs):
        """
        Initializes the BaseModel instance with training and validation data, positive label, and threshold.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
            X_val (array-like): Validation features.
            y_val (array-like): Validation labels.
            pos_label (int, optional): The positive class label. Defaults to 1.
            threshold (float, optional): The classification threshold. Defaults to 0.5.
            enricher (object, optional): Feature enricher instance. Defaults to None.
            enable_cv (bool, optional): Whether to enable cross-validation. Defaults to False.
            cv_params (dict, optional): Cross-validation parameters. Defaults to None.
        """
        self.enricher = enricher
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.pos_label, self.threshold = pos_label, threshold
        self.model = None
        self.model_metrics = None
        self.enable_cv = enable_cv
        self.cv_params = cv_params or {}
        self.cv_results = None

    def fit(self):
        """
        Trains the model. This method should be implemented in child classes.

        Raises:
            NotImplementedError: If not implemented in a child class.
        """
        raise NotImplementedError("The fit() method must be implemented in a child class.")

    def cross_validate(self):
        """
        Runs cross-validation if enabled before training.
        """
        if self.enable_cv:
            print("Running cross-validation before training...")
            cv = CrossValidator(self.model, self.X_train, self.y_train, **self.cv_params)
            self.cv_results = cv.run()

    def predict(self):
        """
        Makes predictions on both training and validation sets.

        Returns:
            BaseModel: The current instance with updated predictions and probabilities.
        """
        self.train_prediction = self.model.predict(self.X_train)
        self.val_prediction = self.model.predict(self.X_val)

        if hasattr(self.model, "predict_proba"):
            self.train_pr_proba = self.model.predict_proba(self.X_train)[:, self.pos_label]
            self.val_pr_proba = self.model.predict_proba(self.X_val)[:, self.pos_label]
        else:
            self.train_pr_proba = self.train_prediction
            self.val_pr_proba = self.val_prediction

        self.__get_metrix()
        return self

    def compute_all_metrics(self):
        """
        Computes all performance metrics for the model.

        Returns:
            list: A list of computed metrics as dictionaries.
        """
        metrics = self.model_metrics.compute_all_metrics()
        return [metric.__dict__ for metric in metrics]

    def display_confusion_matrix(self):
        """
        Displays the confusion matrix for both training and validation sets.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        display_confusion_matrix(self.y_train, self.train_prediction, "Train", axes[0])
        display_confusion_matrix(self.y_val, self.val_prediction, "Validation", axes[1])
        plt.tight_layout()
        plt.show()

    def display_roc_auc(self):
        """
        Displays the ROC-AUC curve for the training and validation sets.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_roc_auc(self.y_train, self.train_pr_proba, self.pos_label, "Train", axes[0])
        plot_roc_auc(self.y_val, self.val_pr_proba, self.pos_label, "Validation", axes[1])
        plt.tight_layout()
        plt.show()

    def display_pr_auc(self):
        """
        Displays the Precision-Recall AUC curve for the training and validation sets.
        """
        fig, ax = plt.subplots(figsize=(6, 5))

        plot_pr_auc(self.y_train, self.train_pr_proba, "Train", ax=ax)
        plot_pr_auc(self.y_val, self.val_pr_proba, "Validation", ax=ax)

        plt.show()

    def __get_metrix(self):
        """
        Returns the ModelEvaluation object, calculating it if necessary.

        Returns:
            ModelEvaluation: An instance containing all metrics related to model performance.
        """
        if self.model_metrics is None:
            self.model_metrics = ModelEvaluation(self.y_train, self.y_val,
                                                 train_prediction=self.train_prediction,
                                                 val_prediction=self.val_prediction,
                                                 train_pr_proba=self.train_pr_proba,
                                                 val_pr_proba=self.val_pr_proba)
        return self.model_metrics

    def evaluate(self):
        """
        Evaluates model performance by plotting residuals for both training and validation sets.
        """
        plot_residuals(self.y_train, self.train_prediction, name="Train set")
        plot_residuals(self.y_val, self.val_prediction, name="Validation set")
