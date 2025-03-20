import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_category_distribution(df: pd.DataFrame, column: str) -> None:
    """
    Plots the distribution of a categorical variable as a bar chart.

    Args:
    df : pd.DataFrame
        The dataframe containing the categorical variable.
    column : str
        The name of the categorical column in the dataframe.

    Returns: None
        Displays the bar chart of the category distribution.
    """
    category_distribution = df[column].value_counts(normalize=True).reset_index()
    category_distribution.columns = [column, "proportion"]

    plt.figure(figsize=(6, 4))
    sns.barplot(data=category_distribution, x=column, y="proportion",
                hue=column, dodge=False, legend=False)

    plt.xlabel(column)
    plt.ylabel("Proportion")
    plt.title(f"Distribution of {column}")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    for i, v in enumerate(category_distribution["proportion"]):
        plt.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=12)

    plt.show()


def plot_histograms(df, numeric_cols, rows=None, cols=2, bins=20):
    """
    Plots histograms for multiple numeric columns to visualize their distributions.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        numeric_cols (list): List of numerical columns to plot.
        rows (int, optional): Number of rows in the subplot grid. If None, it is auto-calculated based on the number of columns.
        cols (int, optional): Number of columns in the subplot grid. Default is 2.
        bins (int, optional): Number of bins for the histogram. Default is 20.

    Returns:
        None: The function generates and displays a grid of histograms.

    Example:
        plot_histograms(df, ['col1', 'col2', 'col3'], bins=15)

    The function generates a set of histograms, each showing the distribution of values for the specified
    numerical columns. The histograms include Kernel Density Estimation (KDE) curves for smoother distribution visualization.

    Notes:
        - If the number of specified columns exceeds the available subplot space, the extra subplots will be removed.
        - The histograms are displayed in a grid layout with the specified number of columns.
        - The color of the histogram bars is set to "royalblue".
    """

    if rows is None:
        rows = -(-len(numeric_cols) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.ravel(axes)

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=bins, kde=True, ax=axes[i], color="royalblue")
        axes[i].set_title(f"Distribution of {col}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, numeric_cols, threshold=0.3, figsize=(15, 8), title="Correlation Heatmap"):
    """
    Plots a correlation heatmap for the given numeric columns, displaying only correlations above the specified threshold.

    Args:
        df (pd.DataFrame): The dataframe containing numeric data.
        numeric_cols (list): List of numerical columns to calculate correlations.
        threshold (float, optional): Minimum absolute correlation value to display. Only correlations with an absolute
                                     value greater than or equal to this threshold will be shown. Default is 0.3.
        figsize (tuple, optional): Figure size. Default is (15, 8).
        title (str, optional): Title for the heatmap. Default is "Correlation Heatmap".

    Returns:
        None: The function generates and displays a heatmap of correlations.

    Example:
        plot_correlation_heatmap(df, ['col1', 'col2', 'col3'], threshold=0.5)

    The function generates a heatmap displaying the Pearson correlation coefficients between the numerical columns.
    Only correlations with an absolute value greater than or equal to the specified threshold are shown.

    Notes:
        - The heatmap uses a green-to-red color palette, with higher correlations represented by green and lower correlations by red.
        - Correlation values are annotated on the heatmap with two decimal points.
    """

    correlation_df = df[numeric_cols].corr(method='pearson')
    mask = (abs(correlation_df) >= threshold) & (abs(correlation_df) < 0.9999)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data=correlation_df[mask], ax=ax, annot=True, cmap="RdYlGn", fmt=".2f")
    ax.set_title(title)
    plt.show()


def plot_boxplots(df, numeric_cols, rows=None, cols=2):
    """
    Plots boxplots for multiple numeric columns to visualize the distribution and identify potential outliers.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        numeric_cols (list): List of numerical columns to plot.
        rows (int, optional): Number of rows in the subplot grid. If None, it is auto-calculated based on the number of columns.
        cols (int, optional): Number of columns in the subplot grid. Default is 2.

    Returns:
        None: The function generates and displays a grid of boxplots.

    Example:
        plot_boxplots(df, ['col1', 'col2', 'col3'])

    The function generates a set of boxplots, each showing the distribution of values in the specified
    numerical columns. Boxplots are helpful for visualizing the spread of the data and identifying any outliers.

    Notes:
        - The function automatically calculates the number of rows required for the subplot grid if not provided.
        - Each boxplot will display the data distribution, including the median, quartiles, and any outliers.
        - The boxplots are displayed in a grid layout with the specified number of columns.
    """

    if rows is None:
        rows = -(-len(numeric_cols) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.ravel(axes)

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i], color="skyblue")
        axes[i].set_title(f"Boxplot: {col}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_combined_boxplot(df, numeric_cols):
    """
    Plots a combined boxplot for multiple numerical columns in a single graph.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        numeric_cols (list): List of numerical columns to plot.

    Returns:
        None: The function generates and displays a boxplot for the specified columns.

    Example:
        plot_combined_boxplot(df, ['col1', 'col2', 'col3'])

    The function generates a boxplot that visualizes the distribution, median,
    and potential outliers for the specified numerical columns in a single plot.

    Notes:
        - The x-axis will represent the columns specified in `numeric_cols`, and the y-axis will represent the values.
        - The x-axis labels will be rotated by 45 degrees for better readability.
        - The function ensures the plot layout is tight for better presentation.
    """

    plt.figure(figsize=(30, 6))
    df[numeric_cols].boxplot(rot=45)
    plt.tight_layout()
    plt.show()


def plot_kde(df0, df1, numeric_cols, rows=None, cols=2):
    """
    Plots Kernel Density Estimate (KDE) plots for numerical columns in two dataframes (representing two classes).
    Each KDE plot visualizes the distribution of values for a specified numeric column, comparing class 0 and class 1.

    Args:
        df0 (pd.DataFrame): The dataframe representing class 0.
        df1 (pd.DataFrame): The dataframe representing class 1.
        numeric_cols (list): List of numerical columns to plot.
        rows (int, optional): Number of rows in the subplot grid. If None, it is auto-calculated based on the number of columns.
        cols (int, optional): Number of columns in the subplot grid. Default is 2.

    Returns:
        None: The function generates and displays a grid of KDE plots.

    Example:
        plot_kde(df0, df1, ['col1', 'col2', 'col3'])

    The function generates a set of KDE plots, each showing the distribution of values in the specified
    numerical columns for both class 0 and class 1. The plots will have overlaid density curves for each class.

    Notes:
        - The function automatically calculates the number of rows required for the subplot grid if not provided.
        - Each KDE plot will have a legend indicating which class the curve corresponds to.
        - The colors used are red for class 0 and blue for class 1, with transparency (alpha) applied for clarity.
    """

    if rows is None:
        rows = -(-len(numeric_cols) // cols)  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.ravel(axes)

    for i, col in enumerate(numeric_cols):
        sns.kdeplot(df0[col], ax=axes[i], label="Class 0", fill=True, color="red", alpha=0.5)
        sns.kdeplot(df1[col], ax=axes[i], label="Class 1", fill=True, color="blue", alpha=0.5)
        axes[i].set_title(f"KDE plot: {col}")
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def visualise_distribution_sub_plot(df, columns, rows=2, cols=2, title=""):
    """
    Plots count plots for categorical columns in a grid layout, showing the distribution of
    values for each specified column.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): List of categorical columns to plot, each representing a group whose value distribution will be visualized.
        rows (int, optional): Number of rows in the subplot grid. Default is 2.
        cols (int, optional): Number of columns in the subplot grid. Default is 2.
        title (str, optional): Title for the entire plot, displayed above the subplots.

    Returns:
        None: The function generates and displays a grid of count plots.

    Example:
        visualise_distribution_sub_plot(df, ['category1', 'category2', 'category3'])

    The function generates a set of count plots, each showing the distribution of values in the specified
    categorical columns. The plot includes annotations for the count of each category. The title of the entire
    plot is set to "Distribution of values for the category: {title}".

    Notes:
        - If the number of specified columns exceeds the available subplot space, the extra subplots will be removed.
        - Each plot will have the x-axis labeled with the categorical column values, and the y-axis will represent the count.
        - The function rotates the x-axis labels by 45 degrees for better visibility.
    """

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    fig.suptitle(f"Distribution of values from a category '{title}'")

    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        plot = sns.countplot(
            x=df[col], order=df[col].value_counts().index,
            ax=ax, hue=df[col], legend=False
        )
        ax.set_title(f"Distribution '{col}'", fontsize=12)
        ax.set_ylabel("Amount")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

        for p in plot.patches:
            height = p.get_height()
            ax.annotate(f'{height}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, color='black',
                        xytext=(0, 5), textcoords='offset points')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def proc_pas_visualise_subplot(df, columns, rows=1, cols=3, title="", positive_label="yes"):
    """
    Plots bar charts showing the percentage of 'yes' responses for each category in the target column
    across multiple categorical columns, presented as subplots in a grid.

    Args:
        positive_label:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list): List of categorical columns to analyze, each representing a group for which
                        the percentage of 'yes' responses will be visualized.
        rows (int, optional): Number of rows in the subplot grid. Default is 1.
        cols (int, optional): Number of columns in the subplot grid. Default is 3.
        title (str, optional): Title for the entire plot, displayed above the subplots.
        positive_label (str): The label of the positive class. Defailt is "yes"

    Returns:
        None: The function generates and displays a grid of bar plots.

    Example:
        proc_yes_visualise_subplot(df, ['category1', 'category2', 'category3'])

    The function generates a set of bar plots, each showing the percentage of 'yes' responses
    for each category in the specified columns. The plot includes labels for the percentage on each bar.
    The title of the entire plot is set to "Percentage of 'yes' responses for the category: {title}".

    Notes:
        - If the number of columns exceeds the number of subplots available, the remaining subplots will be hidden.
        - Each plot will have the x-axis labeled with the categorical column, and the y-axis will show the percentage.
        - The function will rotate the x-axis labels by 45 degrees for better visibility.
    """

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    fig.suptitle(f"Percentage of positive target variable value for category '{title}'")

    axes = axes.flatten()
    for i in range(len(axes)):
        if i < len(columns):
            col = columns[i]
            crosstab = pd.crosstab(df[col], df["y"], normalize="index") * 100
            plot = sns.barplot(data=crosstab.reset_index(), x=col, y=positive_label,
                               order=df[col].value_counts().index, ax=axes[i],
                               hue=col, legend=False)
            axes[i].set_ylabel("Percentage (%)")
            axes[i].set_xlabel(col)
            axes[i].tick_params(axis="x", rotation=45)

            for p in plot.patches:
                axes[i].annotate(f'{p.get_height():.1f}%',
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='center', va='center',
                                 fontsize=10, color='black',
                                 xytext=(0, 10), textcoords='offset points')
        else:
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def bi_cat_countplot(df, column, hue_column, sub_title=""):
    """
    Creates a bar plot showing the normalized distribution of values for a categorical column,
    grouped by another categorical column (hue_column). The plot displays the proportions
    of each category in the specified column, differentiated by the hue_column.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        column (str): The name of the categorical column for which the distribution is plotted.
        hue_column (str): The name of the column used for grouping the data (hue).

    Returns:
        None: The function generates and displays a bar plot.

    Example:
        bi_cat_countplot(df, 'category_column', 'hue_column')

    The plot will show the normalized distribution (in percentage) of the values in `column`,
    grouped by the unique values in `hue_column`. Each bar will represent a category in the
    `column`, and bars will be colored according to the `hue_column` categories.

    Notes:
        - The plot title will be: "Normalized distribution of values by category: {column}".
        - Percentages are rounded to two decimal places.
        - Bars are labeled with the percentage values.
    """

    unique_hue_values = df[hue_column].unique()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(25, 6)

    plt_name = f'Normalized distribution of values by category: {column}. {sub_title}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions * 100).round(2)

    palette = sns.color_palette("tab20", len(unique_hue_values))

    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=ax, title=plt_name, width=0.8, color=palette)

    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%')
