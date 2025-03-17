from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

class ClassDistributionVisualizer:
    """
    A class for visualizing class distribution using PCA-based dimensionality reduction.

    Attributes:
        class_color_map (dict): A mapping of class labels to colors for visualization.
    """

    def __init__(self, target):
        """
        Initializes the visualizer by assigning colors to unique class labels.

        Args:
            target (pd.Series or np.ndarray): The array of target values.
        """
        unique_classes = target.unique()
        colors = plt.cm.tab10(range(len(unique_classes)))
        self.class_color_map = {cls: color for cls, color in zip(unique_classes, colors)}

    def plot_data_with_pca(self, X, y, title: str):
        """
        Applies PCA to reduce dimensionality to 2 components and plots the class distribution.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray): Target labels.
            title (str): The title of the plot.
        """
        X_pca = PCA(n_components=2).fit(X).transform(X)
        fig, ax = plt.subplots(figsize=(15, 4))

        for cls in sorted(set(y)):
            plt.scatter(X_pca[y == cls, 0], X_pca[y == cls, 1],
                        label=f'Class {cls}', alpha=0.6, edgecolor='black', s=30)

        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.show()
        print(y.value_counts())
