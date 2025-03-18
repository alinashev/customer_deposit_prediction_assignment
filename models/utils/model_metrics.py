import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_curve, auc, precision_score, recall_score, \
    precision_recall_curve, balanced_accuracy_score

from customer_deposit_prediction_assignment.utils.config import METRIC_CONFIG


class Metric:
    """
    Represents a single evaluation metric for a model.

    Attributes:
        name (str): The name of the metric (e.g., 'accuracy', 'f1_score').
        train_value (float): The value of the metric for the training data.
        val_value (float): The value of the metric for the validation data.
    """

    def __init__(self, name: str, train_value: float, val_value: float):
        """
        Initializes a Metric instance.

        Args:
            name (str): The name of the metric.
            train_value (float): The value of the metric for the training data.
            val_value (float): The value of the metric for the validation data.
        """
        self.name = name
        self.train_value = train_value
        self.val_value = val_value


class ModelEvaluation:
    """
    A class for calculating and storing performance metrics for machine learning models.

    Attributes:
        y_train (array-like): The true labels for the training data.
        y_val (array-like): The true labels for the validation data.
        pos_label (int): The label for the positive class (default is 1).
        threshold (float): The threshold used to classify predictions as the positive class (default is 0.5).
        metrics (dict): A dictionary to store computed metrics.
        train_prediction (array-like): The predicted labels for the training data.
        val_prediction (array-like): The predicted labels for the validation data.
        train_pr_proba (array-like): The predicted probabilities for the positive class for the training data.
        val_pr_proba (array-like): The predicted probabilities for the positive class for the validation data.
    """

    def __init__(self, y_train, y_val, pos_label=1, threshold=0.5,
                 train_prediction=None, val_prediction=None, train_pr_proba=None, val_pr_proba=None):
        """
        Initializes the ModelMetrics instance with true labels, predicted labels, probabilities, and other settings.

        Args:
            y_train (array-like): The true labels for the training data.
            y_val (array-like): The true labels for the validation data.
            pos_label (int): The label for the positive class (default is 1).
            threshold (float): The threshold used to classify predictions as positive (default is 0.5).
            train_prediction (array-like): The predicted labels for the training data.
            val_prediction (array-like): The predicted labels for the validation data.
            train_pr_proba (array-like): The predicted probabilities for the training data.
            val_pr_proba (array-like): The predicted probabilities for the validation data.
        """
        self.y_train = y_train
        self.y_val = y_val
        self.pos_label = pos_label
        self.threshold = threshold
        self.metrics = []

        self.train_prediction = train_prediction
        self.val_prediction = val_prediction
        self.train_pr_proba = train_pr_proba
        self.val_pr_proba = val_pr_proba
        self.metric_config = METRIC_CONFIG

    def accuracy(self):
        """
        Computes the accuracy metric.

        Returns:
            Metric: The computed accuracy metric.
        """
        accuracy_train = accuracy_score(self.y_train, self.train_prediction)
        accuracy_val = accuracy_score(self.y_val, self.val_prediction)
        metric = Metric("accuracy", accuracy_train, accuracy_val)
        self.metrics.append(metric)
        return metric

    def f1_score_metric(self):
        """
        Computes the F1-score metric.

        Returns:
            Metric: The computed F1-score metric.
        """
        f1_train = f1_score(self.y_train, self.train_prediction)
        f1_val = f1_score(self.y_val, self.val_prediction)
        metric = Metric("f1_score", f1_train, f1_val)
        self.metrics.append(metric)
        return metric

    def precision(self):
        """
        Computes the precision metric.

        Returns:
            Metric: The computed precision metric.
        """
        precision_train = precision_score(self.y_train, (self.train_pr_proba >= self.threshold).astype(int)),
        precision_val = precision_score(self.y_val, (self.val_pr_proba >= self.threshold).astype(int))
        metric = Metric("precision", precision_train, precision_val)
        self.metrics.append(metric)
        return metric

    def recall(self):
        """
        Computes the recall metric.

       Returns:
           Metric: The computed recall metric.
        """
        recall_train = recall_score(self.y_train, (self.train_pr_proba >= self.threshold).astype(int)),
        recall_val = recall_score(self.y_val, (self.val_pr_proba >= self.threshold).astype(int))
        metric = Metric("recall", recall_train, recall_val)
        self.metrics.append(metric)
        return metric

    def roc_auc(self):
        """
        Computes the ROC-AUC metric.

        Returns:
            Metric: The computed ROC-AUC metric.
        """
        fpr_t, tpr_t, _ = roc_curve(self.y_train, self.train_pr_proba)
        fpr_v, tpr_v, _ = roc_curve(self.y_val, self.val_pr_proba)

        roc_auc_train = auc(fpr_t, tpr_t)
        roc_auc_valid = auc(fpr_v, tpr_v)

        metric = Metric("roc_auc", roc_auc_train, roc_auc_valid)
        self.metrics.append(metric)
        return metric

    def pr_auc(self):
        """
        Computes the Precision-Recall AUC metric.

        Returns:
            Metric: The computed PR-AUC metric.
        """
        prec_t, rec_t, _ = precision_recall_curve(self.y_train, self.train_pr_proba)
        prec_v, rec_v, _ = precision_recall_curve(self.y_val, self.val_pr_proba)

        pr_auc_train = auc(rec_t, prec_t)
        pr_auc_valid = auc(rec_v, prec_v)

        metric = Metric("pr_auc", pr_auc_train, pr_auc_valid)
        self.metrics.append(metric)
        return metric

    def balanced_accuracy(self):
        """
        Computes the balanced accuracy metric.

        Returns:
            Metric: The computed balanced accuracy metric.
        """
        y_train_pred = (self.train_pr_proba >= 0.5).astype(int)
        y_val_pred = (self.val_pr_proba >= 0.5).astype(int)

        balanced_acc_train = balanced_accuracy_score(self.y_train, y_train_pred)
        balanced_acc_valid = balanced_accuracy_score(self.y_val, y_val_pred)

        metric = Metric("balanced_accuracy", balanced_acc_train, balanced_acc_valid)
        self.metrics.append(metric)
        return metric

    def g_mean(self):
        """
        Computes the geometric mean metric.

        Returns:
            Metric: The computed G-mean metric.
        """
        y_train_pred = (self.train_pr_proba >= 0.5).astype(int)
        y_val_pred = (self.val_pr_proba >= 0.5).astype(int)

        g_mean_train = geometric_mean_score(self.y_train, y_train_pred)
        g_mean_valid = geometric_mean_score(self.y_val, y_val_pred)

        metric = Metric("g_mean", g_mean_train, g_mean_valid)
        self.metrics.append(metric)
        return metric

    def loss(self, target, prediction):
        """
        Calculates the loss (root mean squared error) between the true and predicted values.

        Args:
            target (array-like): The true values.
            prediction (array-like): The predicted values.

        Returns:
            float: The root mean squared error (RMSE) between the true and predicted values.
        """
        return np.sqrt(mean_squared_error(target, prediction))

    def residuals(self, target, prediction):
        """
        Calculates the residuals (the differences between true values and predicted values).

        Args:
            target (array-like): The true values.
            prediction (array-like): The predicted values.

        Returns:
            array-like: The residuals (differences between true and predicted values).
        """
        return target - prediction

    def compute_all_metrics(self):
        """
        Computes and stores all enabled metrics as per the configuration.

        Returns:
            list: A list of computed Metric objects.
        """
        self.metrics = []
        metric_functions = {
            "accuracy": self.accuracy,
            "f1_score": self.f1_score_metric,
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "pr_auc": self.pr_auc,
            "g_mean": self.g_mean,
            "balanced_accuracy": self.balanced_accuracy
        }

        for metric_name, func in metric_functions.items():
            if self.metric_config.get(metric_name, "disabled") in ["primary", "secondary"]:
                func()
        return self.metrics
