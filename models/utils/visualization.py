import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def plot_roc_curve(fpr, tpr, roc_auc, name):
    """Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (array-like): False Positive Rates.
        tpr (array-like): True Positive Rates.
        roc_auc (float): Area Under the Curve (AUC) value.
        name (str): Name of the dataset or model for the plot title.
    """
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def display_confusion_matrix(y_true, y_pred, dataset_name, ax=None):
    """Displays the normalized confusion matrix as a heatmap.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        dataset_name (str): Name of the dataset for the plot title.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on.
            If None, a new figure is created.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    if ax is None:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
        plt.xlabel('Prediction')
        plt.ylabel('Target')
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.show()
    else:
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Target')
        ax.set_title(f'Confusion Matrix - {dataset_name}')


def plot_residuals(target, prediction, name=""):
    """Visualizes prediction residuals with a scatter plot and histogram.

    Args:
        target (array-like): Ground truth values.
        prediction (array-like): Predicted values.
        name (str, optional): Title for the entire plot. Defaults to "".
    """
    residuals = target - prediction

    plt.figure(figsize=(6, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(target, prediction, alpha=0.6, color="blue")
    plt.plot([min(target), max(target)], [min(target), max(target)], color="red", linestyle="--")
    plt.title(f"Графік розсіювання")
    plt.xlabel("Реальні значення")
    plt.ylabel("Передбачення")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, bins=30, color="blue", label="Залишки")
    plt.title("Гістограма залишків")
    plt.xlabel("Залишки")
    plt.ylabel("Частота")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.suptitle(name, fontsize=12)
    plt.show()


def plot_roc_auc(y_true, y_proba, pos_label, dataset_name, ax=None):
    """Computes and plots the ROC curve with AUC for classification predictions.

    Args:
        y_true (array-like): True binary labels.
        y_proba (array-like): Predicted probabilities for the positive class.
        pos_label (int or str): The class considered as positive.
        dataset_name (str): Name of the dataset for the plot title.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure is created.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    print(f'AUROC for {dataset_name} dataset: {roc_auc:.4f}')
    if ax is None:
        plt.figure(figsize=(6, 5))
        plot_roc_curve(fpr, tpr, roc_auc, dataset_name)
    else:
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {dataset_name}')
        ax.legend(loc='lower right')


def plot_pr_auc(y_true, y_proba, dataset_name, ax=None):
    """Computes and plots the Precision-Recall (PR) curve with PR-AUC.

    Args:
        y_true (array-like): True binary labels.
        y_proba (array-like): Predicted probabilities for the positive class.
        dataset_name (str): Name of the dataset for the plot title.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure is created.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    print(f'PR-AUC for {dataset_name} dataset: {pr_auc:.4f}')

    if ax is None:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()

    ax.plot(recall, precision, lw=2, label=f'PR curve ({dataset_name}, PR-AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid()
