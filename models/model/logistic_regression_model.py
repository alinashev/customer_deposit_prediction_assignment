from sklearn.linear_model import LogisticRegression
from models.model.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, pos_label=1, threshold=0.5, enricher=None, **kwargs):
        self.enricher = enricher
        X_train_transformed = self.enricher.fit_transform(X_train) if self.enricher else X_train
        X_val_transformed = self.enricher.transform(X_val) if self.enricher else X_val

        super().__init__(X_train_transformed, y_train, X_val_transformed, y_val, pos_label, threshold, **kwargs)
        self.model = LogisticRegression(solver='liblinear', **kwargs)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        return self
