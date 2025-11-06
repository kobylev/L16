"""
Classification Module - k-NN Operations
Handles k-NN classification with different label sets
"""

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def train_knn_on_kmeans(train_vectors, kmeans_labels, n_neighbors=5):
    """
    Train k-NN classifier using K-Means cluster labels.

    Args:
        train_vectors: Training vectors array
        kmeans_labels: K-Means cluster labels (integers)
        n_neighbors: Number of neighbors for k-NN (default: 5)

    Returns:
        KNeighborsClassifier: Trained k-NN model
    """
    knn_kmeans = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_kmeans.fit(train_vectors, kmeans_labels)
    return knn_kmeans


def train_knn_on_manual_labels(train_vectors, manual_labels, n_neighbors=5):
    """
    Train k-NN classifier using manual labels.

    Args:
        train_vectors: Training vectors array
        manual_labels: Manual category labels
        n_neighbors: Number of neighbors for k-NN (default: 5)

    Returns:
        KNeighborsClassifier: Trained k-NN model
    """
    knn_manual = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_manual.fit(train_vectors, manual_labels)
    return knn_manual


def predict_with_kmeans_labels(knn_model, test_vectors, cluster_mapping):
    """
    Predict test samples using k-NN trained on K-Means labels.

    Args:
        knn_model: Trained k-NN model (on K-Means labels)
        test_vectors: Test vectors array
        cluster_mapping: Mapping from cluster numbers to Greek letters

    Returns:
        tuple: (integer_predictions, greek_predictions)
    """
    predictions = knn_model.predict(test_vectors)
    greek_predictions = [cluster_mapping[label] for label in predictions]
    return predictions, greek_predictions


def predict_with_manual_labels(knn_model, test_vectors):
    """
    Predict test samples using k-NN trained on manual labels.

    Args:
        knn_model: Trained k-NN model (on manual labels)
        test_vectors: Test vectors array

    Returns:
        array: Predicted labels
    """
    return knn_model.predict(test_vectors)


def find_nearest_neighbors(train_vectors, test_vector, n_neighbors=5):
    """
    Find the k nearest neighbors for a test vector.

    Args:
        train_vectors: Training vectors array
        test_vector: Single test vector (1D array)
        n_neighbors: Number of neighbors to find (default: 5)

    Returns:
        tuple: (distances, indices) of nearest neighbors
    """
    knn_finder = NearestNeighbors(n_neighbors=n_neighbors)
    knn_finder.fit(train_vectors)
    distances, indices = knn_finder.kneighbors(test_vector.reshape(1, -1))
    return distances, indices
