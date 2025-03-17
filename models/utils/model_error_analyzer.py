import seaborn as sns
import matplotlib.pyplot as plt


class ModelErrorAnalyzer:
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.y_pred = model.val_prediction  # Передбачені класи
        self.y_pred_proba = model.val_pr_proba  # Передбачені ймовірності

    def get_misclassified_samples(self):
        false_positives = self.X_val[(self.y_val == 0) & (self.y_pred == 1)].copy()
        false_negatives = self.X_val[(self.y_val == 1) & (self.y_pred == 0)].copy()

        # Додаємо реальні значення, передбачені класи та ймовірності
        false_positives["y_real"] = 0
        false_positives["y_pred"] = 1
        false_positives["y_pred_proba"] = self.y_pred_proba[(self.y_val == 0) & (self.y_pred == 1)]

        false_negatives["y_real"] = 1
        false_negatives["y_pred"] = 0
        false_negatives["y_pred_proba"] = self.y_pred_proba[(self.y_val == 1) & (self.y_pred == 0)]

        return false_positives, false_negatives

    def visualize_misclassified_samples(self):
        false_positives, false_negatives = self.get_misclassified_samples()

        plt.figure(figsize=(12, 5))
        sns.histplot(false_positives["y_pred_proba"], bins=20, kde=True, color='blue', label="False Positives")
        sns.histplot(false_negatives["y_pred_proba"], bins=20, kde=True, color='red', label="False Negatives",
                     alpha=0.7)
        plt.xlabel("Ймовірність передбачення")
        plt.ylabel("Кількість записів")
        plt.title("Розподіл ймовірностей у помилкових передбаченнях")
        plt.legend()
        plt.show()

    def analyze_top_features(self, top_n=10):
        false_positives, false_negatives = self.get_misclassified_samples()
        df_val_means = self.X_val.mean()

        fp_diff = (false_positives.mean() - df_val_means).abs().sort_values(ascending=False).head(top_n)
        fn_diff = (false_negatives.mean() - df_val_means).abs().sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(y=fp_diff.index, x=fp_diff.values, color="blue", label="False Positives")
        sns.barplot(y=fn_diff.index, x=fn_diff.values, color="red", label="False Negatives", alpha=0.7)
        plt.xlabel("Відхилення від середнього")
        plt.ylabel("Ознаки")
        plt.title("Найбільш відмінні ознаки у помилкових передбаченнях")
        plt.legend()
        plt.show()
