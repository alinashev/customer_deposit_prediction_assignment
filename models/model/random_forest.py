from sklearn.ensemble import RandomForestClassifier
from models.model.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, enable_cv=False, cv_params=None, **kwargs):
        super().__init__(X_train, y_train, X_val, y_val, enable_cv=enable_cv, cv_params=cv_params, **kwargs)

        self.model = RandomForestClassifier(
            random_state=42,
            **kwargs
        )

    def fit(self):
        self.cross_validate()
        self.model.fit(self.X_train, self.y_train)
        return self
