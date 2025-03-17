import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe, hp
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import roc_auc_score

from tuning.hyperparameter_tuner import HyperparameterTuner


class HyperoptTuner(HyperparameterTuner):
    def objective(self, params):
        model = self.model_class(
            **{k: int(v) if k in ['n_estimators', 'max_depth', 'min_child_weight', 'num_leaves'] else v for k, v in
               params.items()},
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        val_pred = model.predict_proba(self.X_val)[:, 1]
        roc_auc = roc_auc_score(self.y_val, val_pred)
        return {'loss': -roc_auc, 'status': STATUS_OK}

    def tune(self, transform=True):
        trials = Trials()
        if transform:
            space = self.transform_to_hyperopt_space(self.param_space)
        else:
            space = self.param_space

        best_params = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            early_stop_fn = no_progress_loss(7)
        )
        self.best_params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_child_weight', 'num_leaves'] else v for
                            k, v in best_params.items()}
        return self.best_params

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
                    param_space_hyperopt[param] = hp.quniform(param, values[0], values[-1], step)
                else:
                    param_space_hyperopt[param] = hp.uniform(param, values[0], values[-1])
            else:
                param_space_hyperopt[param] = hp.uniform(param, values[0], values[-1])

        return param_space_hyperopt
