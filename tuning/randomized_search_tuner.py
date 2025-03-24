from sklearn.model_selection import RandomizedSearchCV

from customer_deposit_prediction_assignment.tuning.hyperparameter_tuner import HyperparameterTuner


class RandomSearchTuner(HyperparameterTuner):
    """Performs hyperparameter tuning using RandomizedSearchCV.

    This class implements random search tuning based on scikit-learn's `RandomizedSearchCV`.
    It supports both LGBMClassifier (with suppressed verbosity) and other scikit-learn-compatible estimators.

    Inherits from:
        HyperparameterTuner: Base class for hyperparameter tuning.
    """

    def tune(self, n_iter=40, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1):
        """Performs random search to find the best hyperparameters.

        Args:
            n_iter (int): Number of parameter settings that are sampled. Default is 40.
            cv (int or cross-validation generator): Number of cross-validation folds. Default is 3.
            scoring (str or callable): Metric used to evaluate model performance. Default is 'roc_auc'.
            random_state (int): Seed used for reproducibility. Default is 42.
            n_jobs (int): Number of parallel jobs to run. Default is -1 (use all processors).

        Returns:
            dict: The best set of hyperparameters found during the search.
        """

        if self.model_class.__name__ == "LGBMClassifier":
            model = self.model_class(verbosity=-1)
        else:
            model = self.model_class()
            random_search = RandomizedSearchCV(
                estimator=model, param_distributions=self.param_space, n_iter=n_iter,
                cv=cv, scoring=scoring, random_state=random_state, n_jobs=n_jobs
            )
            random_search.fit(self.X_train, self.y_train)
            self.best_params = random_search.best_params_
            return self.best_params
