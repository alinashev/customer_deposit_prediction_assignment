from sklearn.model_selection import RandomizedSearchCV

from tuning.hyperparameter_tuner import HyperparameterTuner


class RandomSearchTuner(HyperparameterTuner):
    def tune(self, n_iter=40, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1):
        model = self.model_class()
        random_search = RandomizedSearchCV(
            estimator=model, param_distributions=self.param_space, n_iter=n_iter,
            cv=cv, scoring=scoring, random_state=random_state, n_jobs=n_jobs
        )
        random_search.fit(self.X_train, self.y_train)
        self.best_params = random_search.best_params_
        return self.best_params