import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

from utils.config import METRIC_CONFIG


class ModelComparison:
    def __init__(self, metric_config=METRIC_CONFIG):
        self.models = []
        self.df = None  # Змінна для збереження DataFrame
        self.console = Console()
        self.model_registry = {}  # Словник для збереження моделей за їхнім хешем

        # Встановлюємо метрики з конфігурації
        self.metric_config = metric_config
        self.primary_metric, self.secondary_metric = self.set_metrics_from_config()

    def set_metrics_from_config(self):
        primary_metric = None
        secondary_metric = None

        for metric, role in self.metric_config.items():
            metric_v_T = f"{metric}_T"  # Версія для тренувальних даних
            metric_v_V = f"{metric}_V"  # Версія для валідаційних даних

            if role == "primary":
                primary_metric = (metric_v_T, metric_v_V)
            elif role == "secondary":
                secondary_metric = metric_v_V  # Другорядна метрика тільки для валідації

        # Переконаємося, що є значення за замовчуванням
        if not primary_metric:
            primary_metric = ("roc_auc_T", "roc_auc_V")
        if not secondary_metric:
            secondary_metric = "f1_score_V"

        return primary_metric, secondary_metric

    def add_model(self, model, notes=""):
        model_name = model.__class__.__name__
        params = model.model.get_params() if hasattr(model.model, "get_params") else {}

        # Приведення значень до стандартних Python-типів
        params = {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
                  for k, v in params.items()}

        # Отримуємо метрики
        metrics_list = model.compute_all_metrics()
        metrics = {
            f"{m['name']}_T": round(m["train_value"], 4)
            for m in metrics_list
        }
        metrics.update({
            f"{m['name']}_V": round(m["val_value"], 4)
            for m in metrics_list
        })

        # Генеруємо унікальний хеш для моделі
        model_hash = hash(model)

        # Додаємо модель у реєстр
        self.model_registry[model_hash] = model

        # Додаємо у список
        self.models.append({
            "Model": model_name,
            "Hyperparameters": params,
            self.primary_metric[0]: metrics.get(self.primary_metric[0], None),  # `roc_auc_T`
            self.primary_metric[1]: metrics.get(self.primary_metric[1], None),  # `roc_auc_V`
            "Notes": notes,
            "Comment": "",  # Нова колонка для коментарів
            "_Hash": model_hash  # Хеш зберігається, але не відображається у фінальному порівнянні
        })

        # Оновлюємо DataFrame після кожного додавання моделі
        self.df = pd.DataFrame(self.models)

    def compare_models(self):
        if not self.models:
            print("[Warning] No models added for comparison.")
            return pd.DataFrame()

        return self.df[["Model", "Hyperparameters", self.primary_metric[0], self.primary_metric[1], "Notes", "Comment"]]

    def update_model_comment(self, model, new_comment):
        model_hash = hash(model)

        if self.df is not None and "_Hash" in self.df.columns:
            mask = self.df["_Hash"] == model_hash

            if mask.sum() == 1:
                # Оновлюємо коментар у DataFrame
                self.df.loc[mask, "Comment"] = new_comment

                # Також оновлюємо значення в `self.models`
                for model_entry in self.models:
                    if model_entry["_Hash"] == model_hash:
                        model_entry["Comment"] = new_comment
                        break
            elif mask.sum() > 1:
                print(f"[Warning] Знайдено кілька записів з Model_Hash={model_hash}, уточніть критерії пошуку.")
        else:
            print(f"[Error] Model_Hash={model_hash} не знайдено.")

    def display_metrics(self, metrics: list):
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
