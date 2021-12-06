# Pure Python
import warnings
from collections import Counter
from typing import Dict, List, Union, Callable
from itertools import permutations
import math

# Numerical
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix

# Plots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns


def _get_ratings_freq(s: pd.Series) -> pd.Series:
    return pd.Series(Counter(s)) / len(s)


def get_ratings_freq(df: pd.DataFrame) -> pd.Series:
    col = "rating"
    return _get_ratings_freq(df[col])


def update_rating(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    max_freq: float = 0.6,
    min_ratings: int = 4,
):
    """
    Updates in-place the ratings using given mapping
    """
    col = "rating"

    assert (
        len(set(mapping.values())) >= min_ratings
    ), f"Mapping should contain at least {min_ratings} different ratings"
    if not all(x in mapping for x in df[col]):
        warnings.warn(
            f"Mapping `{mapping}` does not map every element in df[col]", UserWarning
        )

    def rename(x):
        if x in mapping:
            return mapping[x]
        else:
            return x

    tmp = df[col].apply(rename)

    n = len(tmp)

    assert all(
        (count / n) <= max_freq for _, count in Counter(tmp).items()
    ), f"New rating system has too many elements in at least one of the new categories\n{_get_ratings_freq(tmp)}"

    # Update dataframe in-place
    df[col] = tmp


def pairwise_distance(
    X: np.ndarray, metric: Union[str, Callable] = "euclidean", **kwargs
):
    """
    Return the pairwise distance between observations (rows) in X.

    For more details, read the documentation of `scipy.spatial.distance.pdist`.

    :param X: an m by n array of m original observations in an n-dimensional space
    :param metric: the distance metric to use
    :param kwargs: extra arguments to metric
    """
    return squareform(pdist(X, metric=metric, **kwargs))


def word_cloud(occurences: List[str]):
    """
    Plot a word cloud based on a list of occurences.
    The more an item appears in the list, the bigger it will be displayed.
    """
    freqs = Counter(occurences)
    plt.figure(figsize=(12, 15))
    wc = WordCloud(
        max_words=1000, background_color="white", random_state=1, width=1200, height=600
    ).generate_from_frequencies(freqs)
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def group_visualization(
    target: np.ndarray, X: np.ndarray, less_points: bool = False, save_fig: str = None
):
    """
    Plot points in a scatter format, where groups are colorized differently.

    :param target: ratings
    :param X: coordinates
    :param less_points: if True, only plot a subset
    :param save_fig: if present, will save figure at given location (require the `kaleida` package)
    """
    target = target.reshape(-1, 1)
    dim = X.shape[1]
    assert dim in [2, 3], "Can only deal with 2D or 3D data"

    columns = ["x", "y", "rating"]

    if dim == 3:
        columns.insert(2, "z")

    if less_points:
        indices = np.random.choice(range(X.shape[0]), size=800, replace=False)
        X = X[indices, :]
        target = target[indices, :]

    df = pd.DataFrame(np.column_stack([X, target]), columns=columns)

    if dim == 2:
        fig = px.scatter(df, x="x", y="y", color="rating")
    else:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="rating")

    if save_fig:
        fig.write_image(save_fig)

    fig.show()


def silhouette_visualization(cluster_labels: np.ndarray, distance_matrix: np.ndarray):
    """
    Draw clusters' silhouette, inspired from:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

    :param cluster_labels: prediction of cluster indices
    :param distance matrix: distance matrix from observations
    """
    sns.set()
    colors = iter(
        [
            "gold",
            "mediumaquamarine",
            "midnightblue",
            "red",
            "purple",
            "green",
            "brown",
            "gray",
        ]
    )  # expand if K>8
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    silhouette_avg = silhouette_score(
        distance_matrix, cluster_labels, metric="precomputed"
    )
    sample_silhouette_values = silhouette_samples(
        distance_matrix, cluster_labels, metric="precomputed"
    )
    y_lower = 10

    # Create a subplot with 1 row and 2 columns
    n_clusters = (np.unique(cluster_labels)).shape[0]
    for i in range(n_clusters):

        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=next(colors),
            edgecolor="k",
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("Silhouette of predicted clusters.")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    plt.show()


def confusion_matrix_visualization(
    y_true: np.ndarray, cluster_labels: np.ndarray, cluster_mapping: Dict
):
    """
    Assign labels to clusters, create the confusion matrix between ground truth and prediction, and plot it.
    Source:
    https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap

    :param y_true: true labels
    :param cluster_labels: prediction of cluster indices
    :param cluster_mapping: mapping between cluster indices and labels
    """
    y_pred = np.empty_like(y_true)

    for cluster_label, cluster_name in cluster_mapping.items():
        y_pred[cluster_labels == cluster_label] = cluster_name

    labels = sorted(cluster_mapping.values())

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    z_text = [[y for y in x] for x in matrix]

    fig = ff.create_annotated_heatmap(
        matrix, x=labels, y=labels, annotation_text=z_text, colorscale="Viridis"
    )

    # add title
    fig.update_layout(
        title_text="<i><b>Confusion matrix</b></i>",
        xaxis=dict(title="Prediction"),
        yaxis=dict(title="True value"),
    )

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig["data"][0]["showscale"] = True
    fig.show()
