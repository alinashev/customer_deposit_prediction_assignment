class BaseResampler:
    """
    A base class for implementing resampling techniques to address class imbalance.

    Attributes:
        sampling_strategy (str or float): The strategy for resampling the minority class.
        random_state (int or None): Controls the randomness of resampling.
    """

    def __init__(self, sampling_strategy: str = 'auto', random_state: int = None):
        """
        Initializes the BaseResampler with a sampling strategy and optional random state.

        Args:
            sampling_strategy (str or float, optional): The strategy for resampling. Defaults to 'auto'.
            random_state (int or None, optional): The random seed for reproducibility. Defaults to None.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        Abstract method that must be implemented by subclasses to perform resampling.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")