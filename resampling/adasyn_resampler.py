import pandas as pd
from imblearn.over_sampling import ADASYN

from resampling.resampler import BaseResampler

class AdasynResampler(BaseResampler):
    """
    A class for handling class imbalance using the ADASYN (Adaptive Synthetic) resampling method.

    Attributes:
        sampling_strategy (str or float): The desired ratio of the minority class after resampling.
        random_state (int or None): Controls the randomness of the resampling.
        n_neighbors (int): Number of nearest neighbors to use for synthetic sample generation.
        sampler (ADASYN): The ADASYN sampler instance used for resampling.
    """

    def __init__(self, sampling_strategy: str = 'auto', random_state: int = None, n_neighbors: int = 5):
        """
        Initializes the AdasynResampler with the specified sampling strategy, random state, and neighbors.

        Args:
            sampling_strategy (str or float, optional): The sampling strategy for balancing classes. Defaults to 'auto'.
            random_state (int or None, optional): The random seed for reproducibility. Defaults to None.
            n_neighbors (int, optional): Number of nearest neighbors to consider when generating synthetic samples. Defaults to 5.
        """
        super().__init__(sampling_strategy, random_state)
        self.n_neighbors = n_neighbors
        self.sampler = ADASYN(sampling_strategy=self.sampling_strategy, random_state=self.random_state,
                              n_neighbors=self.n_neighbors)

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Applies ADASYN resampling to balance the class distribution.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.

        Returns:
            tuple: A tuple containing the resampled feature matrix (pd.DataFrame) and target labels (pd.Series).
        """
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
