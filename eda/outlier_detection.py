import numpy as np
import pandas as pd
from scipy.stats import zscore


def detect_outliers_zscore(df, columns, threshold=3):
    """
    Detects outliers in specified numerical columns using the Z-score method.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): List of numerical columns to check for outliers.
        threshold (float, optional): Z-score threshold to classify outliers. Default is 3.

    Returns:
        dict: A dictionary where keys are column names and values are series of outliers
              (rows that exceed the specified Z-score threshold).

    Example:
        outliers = detect_outliers_zscore(df, ['col1', 'col2'], threshold=2.5)

    The function calculates the Z-score for each value in the specified columns and identifies rows
    that have a Z-score greater than the specified threshold (by default, 3). These rows are classified as outliers.

    Notes:
        - The Z-score is computed as the number of standard deviations a data point is away from the mean.
        - Outliers are those values whose absolute Z-score exceeds the threshold value.
    """
    outliers = {}
    for col in columns:
        z_scores = np.abs(zscore(df[col]))
        outliers[col] = df[col][z_scores > threshold]
    return outliers


def detect_outliers_iqr(df, columns):
    """
    Detects outliers in specified numerical columns using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): List of numerical columns to check for outliers.

    Returns:
        dict: A dictionary where keys are column names and values are series of outliers
              (values outside the IQR bounds for each column).

    Example:
        outliers = detect_outliers_iqr(df, ['col1', 'col2'])

    The function calculates the IQR (Interquartile Range) for each specified column and identifies
    values that fall outside the range defined by 1.5 times the IQR below the first quartile
    (Q1) and above the third quartile (Q3). These values are classified as outliers.

    Notes:
        - Outliers are those values that fall below the lower bound (Q1 - 1.5 * IQR) or above the upper bound (Q3 + 1.5 * IQR).
        - The IQR method is commonly used to detect potential outliers in a dataset.
    """
    outliers = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return outliers
