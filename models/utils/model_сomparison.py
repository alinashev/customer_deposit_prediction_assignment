import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

from customer_deposit_prediction_assignment.utils.config import METRIC_CONFIG


class ModelComparison:
    """Class for registering and comparing multiple ML models based on evaluation metrics.

    This class allows adding models along with their hyperparameters, computed metrics, and notes.
    It provides functionality to compare models using primary and secondary metrics from a
    configuration, update model comments, and display metric tables.

    Attributes:
        models (list): A list of dictionaries, each representing a model and its metadata.
        df (pd.DataFrame): DataFrame that stores the models and associated information.
        console (Console): Rich console object used for rendering tables.
        model_registry (dict): Dictionary mapping model hashes to model instances.
        metric_config (dict): Dictionary defining the primary and secondary metrics.
        primary_metric (tuple): Tuple of (train_metric, val_metric) used as primary comparison.
        secondary_metric (str): Validation metric used as a secondary comparison.
    """

    def __init__(self, metric_config=METRIC_CONFIG):
        """Initializes the ModelComparison object.

        Args:
            metric_config (dict): Dictionary defining metric roles, e.g., {"roc_auc": "primary"}.
        """
        self.models = []
        self.df = None
        self.console = Console()
        self.model_registry = {}
        self.metric_config = metric_config
        self.primary_metric, self.secondary_metric = self.set_metrics_from_config()

    def set_metrics_from_config(self):
        """Parses the metric configuration and sets primary and secondary metrics.

        Returns:
            tuple: A tuple (primary_metric_train, primary_metric_val) and secondary_metric_val.
        """
        primary_metric = None
        secondary_metric = None

        for metric, role in self.metric_config.items():
            metric_v_T = f"{metric}_T"
            metric_v_V = f"{metric}_V"

            if role == "primary":
                primary_metric = (metric_v_T, metric_v_V)
            elif role == "secondary":
                secondary_metric = metric_v_V

        if not primary_metric:
            primary_metric = ("roc_auc_T", "roc_auc_V")
        if not secondary_metric:
            secondary_metric = "f1_score_V"

        return primary_metric, secondary_metric

    def add_model(self, model, notes=""):
        """Adds a trained model to the comparison list with its metrics and notes.

        Args:
            model: A trained model object with `model` and `compute_all_metrics()` method.
            notes (str): Optional notes or context to store with the model.
        """
        model_name = model.__class__.__name__
        params = model.model.get_params() if hasattr(model.model, "get_params") else {}

        params = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
                  for k, v in params.items()}

        metrics_list = model.compute_all_metrics()
        metrics = {
            f"{m['name']}_T": round(m["train_value"], 4)
            for m in metrics_list
        }
        metrics.update({
            f"{m['name']}_V": round(m["val_value"], 4)
            for m in metrics_list
        })

        model_hash = hash(model)
        self.model_registry[model_hash] = model

        self.models.append({
            "Model": model_name,
            "Hyperparameters": params,
            self.primary_metric[0]: metrics.get(self.primary_metric[0], None),
            self.primary_metric[1]: metrics.get(self.primary_metric[1], None),
            "Notes": notes,
            "Comment": "",
            "_Hash": model_hash
        })

        self.df = pd.DataFrame(self.models)

    def compare_models(self):
        """Returns a summary DataFrame with primary metrics for all added models.

        Returns:
            pd.DataFrame: A filtered DataFrame showing key metrics and notes for comparison.
        """
        if not self.models:
            print("[Warning] No models added for comparison.")
            return pd.DataFrame()

        return self.df[["Model", "Hyperparameters", self.primary_metric[0], self.primary_metric[1], "Notes", "Comment"]]

    def update_model_comment(self, model, new_comment):
        """Updates the comment field for a specific model.

        Args:
            model: The model object whose comment needs to be updated.
            new_comment (str): The new comment text.
        """
        model_hash = hash(model)

        if self.df is not None and "_Hash" in self.df.columns:
            mask = self.df["_Hash"] == model_hash

            if mask.sum() == 1:
                self.df.loc[mask, "Comment"] = new_comment
                for model_entry in self.models:
                    if model_entry["_Hash"] == model_hash:
                        model_entry["Comment"] = new_comment
                        break
            elif mask.sum() > 1:
                print(f"[Warning] Multiple entries found for Model_Hash={model_hash}, please disambiguate.")
        else:
            print(f"[Error] Model_Hash={model_hash} not found.")

    def display_metrics(self, metrics: list):
        """Displays a table of evaluation metrics using the rich console.

        Args:
            metrics (list): A list of metric dictionaries with keys: 'name', 'train_value', 'val_value'.
        """
        table = Table(title="Model Evaluation Metrics")
        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Train Value", justify="right", style="green")
        table.add_column("Validation Value", justify="right", style="magenta")

        for metric in metrics:
            table.add_row(
                metric["name"],
                f"{metric['train_value']:.4f}",
                f"{metric['val_value']:.4f}"
            )

        self.console.print(table)
