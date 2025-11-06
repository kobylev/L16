"""
Clustering Module - K-Means Operations
Handles K-Means clustering and cluster analysis
"""

from sklearn.cluster import KMeans
from collections import Counter


def perform_kmeans_clustering(train_vectors, n_clusters=3, random_state=42):
    """
    Perform K-Means clustering on training vectors.

    Args:
        train_vectors: Training vectors array
        n_clusters: Number of clusters (default: 3)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        tuple: (kmeans_model, kmeans_labels, kmeans_labels_greek)
            - kmeans_model: Fitted KMeans object
            - kmeans_labels: Cluster labels as integers
            - kmeans_labels_greek: Cluster labels as Greek letters
    """
    # Run K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(train_vectors)

    # Map cluster numbers to Greek letters for clarity
    cluster_mapping = {0: "α", 1: "β", 2: "γ"}
    kmeans_labels_greek = [cluster_mapping[label] for label in kmeans_labels]

    return kmeans, kmeans_labels, kmeans_labels_greek


def analyze_cluster_alignment(kmeans_labels, manual_labels):
    """
    Analyze alignment between K-Means clusters and manual labels.

    Args:
        kmeans_labels: K-Means cluster labels (integers)
        manual_labels: Manual category labels

    Returns:
        dict: Contains cluster_themes mapping and alignment_accuracy
    """
    cluster_mapping = {0: "α", 1: "β", 2: "γ"}

    # Create a mapping to find most common manual label per cluster
    cluster_analysis = {0: [], 1: [], 2: []}
    for i, cluster in enumerate(kmeans_labels):
        cluster_analysis[cluster].append(manual_labels[i])

    # Find dominant theme in each cluster
    cluster_themes = {}
    cluster_info = {}

    for cluster_num, labels in cluster_analysis.items():
        label_counts = Counter(labels)
        dominant_label = label_counts.most_common(1)[0][0]
        cluster_themes[cluster_num] = dominant_label

        greek = cluster_mapping[cluster_num]

        # Determine theme
        if dominant_label == "A":
            theme = "Hope/Aspiration"
        elif dominant_label == "B":
            theme = "Conflict/Violence"
        else:
            theme = "Science/Technology"

        cluster_info[greek] = {
            'label_counts': dict(label_counts),
            'dominant_label': dominant_label,
            'theme': theme
        }

    return {
        'cluster_themes': cluster_themes,
        'cluster_info': cluster_info,
        'cluster_mapping': cluster_mapping
    }


def get_cluster_distribution(kmeans_labels_greek):
    """
    Get distribution of samples across clusters.

    Args:
        kmeans_labels_greek: Cluster labels as Greek letters

    Returns:
        Counter: Distribution of cluster labels
    """
    return Counter(kmeans_labels_greek)
