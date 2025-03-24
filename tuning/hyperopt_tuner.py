import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe, hp
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import roc_auc_score

from customer_deposit_prediction_assignment.tuning.hyperparameter_tuner import HyperparameterTuner


class HyperoptTuner(HyperparameterTuner):
    """Hyperparameter tuning using the Hyperopt library.

    This class implements Bayesian optimization via the Tree-structured Parzen Estimator (TPE)
    from the Hyperopt library. It tunes the hyperparameters of a given model class to maximize
    the ROC-AUC score on the validation set.

    Inherits from:
        HyperparameterTuner: A base class for hyperparameter tuning.
    """
    def objective(self, params):

        """Objective function to minimize (negative ROC-AUC) during optimization.

        Args:
            params (dict): Dictionary of hyperparameters to evaluate.

        Returns:
            dict: Dictionary containing:
                - 'loss': Negative ROC-AUC score on the validation set.
                - 'status': Optimization status (STATUS_OK).
        """
        params = {k: int(v) if k in ["num_iterations", "n_estimators", "max_depth", "min_child_weight", "num_leaves"]
        else v for k, v in params.items()}

        model = self.model_class(**params, verbosity=-1) if self.model_class.__name__ == "LGBMClassifier" \
            else self.model_class(**params, n_jobs=-1)

        model.fit(self.X_train, self.y_train)
        val_pred = model.predict_proba(self.X_val)[:, 1]
        roc_auc = roc_auc_score(self.y_val, val_pred)
        return {"loss": -roc_auc, "status": STATUS_OK}

    def tune(self, transform=True):
        """Runs the hyperparameter optimization process using Hyperopt.

        Args:
            transform (bool): Whether to transform the parameter space to Hyperopt format.
                Default is True.

        Returns:
            dict: Best set of hyperparameters found, with integer values cast appropriately.
        """
        trials = Trials()
        space = self.transform_to_hyperopt_space(self.param_space) if transform else self.param_space

        best_params = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            early_stop_fn=no_progress_loss(7)
        )

        return {k: int(v) if k in ["num_iterations", "n_estimators", "max_depth", "min_child_weight", "num_leaves"]
        else v for k, v in best_params.items()}

    def transform_to_hyperopt_space(self, param_space_random):
        """
        Converts the parameter space in the form of random values (using np.arange or np.linspace)
        to a format compatible with hyperopt's search space (using hp.quniform and hp.uniform).

        Args:
        - param_space_random (dict): Dictionary containing parameter names as keys and
          their corresponding values as arrays (generated using np.arange or np.linspace).

        Returns:
        - dict: A dictionary where each parameter is mapped to a hyperopt-compatible search space
          (using hp.quniform for discrete ranges and hp.uniform for continuous ranges).
        """
        param_space_hyperopt = {}

        for param, values in param_space_random.items():
            if isinstance(values, np.ndarray):
                if len(values) > 1 and np.allclose(np.diff(values), values[1] - values[0]):
                    step = values[1] - values[0]
                    if param in ["num_iterations", "n_estimators", "max_depth", "min_child_weight", "num_leaves"]:
                        param_space_hyperopt[param] = hp.quniform(param, values[0], values[-1], step)
                    else:
                        param_space_hyperopt[param] = hp.uniform(param, values[0], values[-1])
                else:
                    param_space_hyperopt[param] = hp.uniform(param, values[0], values[-1])
            else:
                param_space_hyperopt[param] = hp.uniform(param, values[0], values[-1])

        return param_space_hyperopt
