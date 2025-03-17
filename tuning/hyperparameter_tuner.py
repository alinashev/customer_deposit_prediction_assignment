class HyperparameterTuner:
    def __init__(self, model_class, X_train, y_train, X_val, y_val, param_space, max_evals=20):
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.param_space = param_space
        self.max_evals = max_evals
        self.best_params = None

    def tune(self):
        raise NotImplementedError("The tune() method must be implemented in a child class.")
