from sklearn.tree import DecisionTreeClassifier
from models.model.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, max_depth=None, criterion="gini",
                 enable_cv=False, cv_params=None, **kwargs):
        super().__init__(X_train, y_train, X_val, y_val, enable_cv=enable_cv, cv_params=cv_params, **kwargs)
        self.model = DecisionTreeClassifier(max_depth=max_depth,
                                            criterion=criterion,
                                            **kwargs)

    def fit(self):
        self.cross_validate()
        self.model.fit(self.X_train, self.y_train)
        return self
