from models.model.base_model import BaseModel

import xgboost as xgb


class XGBModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, params=None, enable_cv=False, cv_params=None, **kwargs):
        super().__init__(X_train, y_train, X_val, y_val, enable_cv=enable_cv, cv_params=cv_params, **kwargs)
        self.params = params if params else {}
        self.kwargs = kwargs
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        return self
