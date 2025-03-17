from sklearn.preprocessing import PolynomialFeatures


class Enricher:
    """
    Base class for feature engineering.

    Methods:
        fit_transform(X): Abstract method that must be implemented by subclasses.
    """

    def fit_transform(self, X):
        """
        Abstract method for transforming features. Must be implemented by subclasses.

        Args:
            X (array-like): Input features.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the fit_transform method.")


class PolynomialFeatureEnricher(Enricher):
    """
    Applies polynomial feature transformation to the input data.

    Attributes:
        degree (int): The degree of polynomial features to generate.
        poly (PolynomialFeatures): The PolynomialFeatures instance for transformation.
    """

    def __init__(self, degree: int = 2, include_bias: bool = False):
        """
        Initializes the PolynomialFeatureEnricher with specified parameters.

        Args:
            degree (int, optional): The polynomial degree. Defaults to 2.
            include_bias (bool, optional): Whether to include a bias column. Defaults to False.
        """
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=include_bias)

    def fit_transform(self, X):
        """
        Fits the polynomial transformer and applies the transformation.

        Args:
            X (array-like): The input features.

        Returns:
            array-like: The transformed feature set with polynomial features.
        """
        return self.poly.fit_transform(X)

    def transform(self, X):
        """
        Applies the polynomial transformation to new data.

        Args:
            X (array-like): The input features to transform.

        Returns:
            array-like: The transformed feature set with polynomial features.
        """
        return self.poly.transform(X)
