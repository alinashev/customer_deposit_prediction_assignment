import pandas as pd
from imblearn.over_sampling import SMOTENC

from resampling.resampler import BaseResampler

class SmotencResampler(BaseResampler):
    """
    A class for handling class imbalance using the SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features) resampling method.

    Attributes:
        sampling_strategy (str or float): The desired ratio of the minority class after resampling.
        random_state (int or None): Controls the randomness of the resampling.
        categorical_features (list[int] or None): List of indices representing categorical features.
        sampler (SMOTENC): The SMOTENC sampler instance used for resampling.
    """

    def __init__(self, sampling_strategy: str = 0.5, random_state: int = 42, categorical_features: list = None):
        """
        Initializes the SmotencResampler with the specified sampling strategy, random state, and categorical feature indices.

        Args:
            sampling_strategy (str or float, optional): The sampling strategy for balancing classes. Defaults to 'auto'.
            random_state (int, optional): The random seed for reproducibility. Defaults to 42.
            categorical_features (list[int] or None, optional): List of categorical feature indices. Defaults to None.
        """
        super().__init__(sampling_strategy, random_state)
        self.categorical_features = categorical_features
        self.sampler = SMOTENC(sampling_strategy=self.sampling_strategy, random_state=self.random_state,
                               categorical_features=self.categorical_features)

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Applies SMOTENC resampling to balance the class distribution while considering categorical features.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.

        Returns:
            tuple: A tuple containing the resampled feature matrix (pd.DataFrame) and target labels (pd.Series).
        """
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
