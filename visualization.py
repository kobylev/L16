"""
Visualization Module - Plotting and Graph Generation
Handles all visualization and plotting operations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from collections import Counter


def generate_main_visualization(
    train_vectors, test_vectors, kmeans_labels, manual_labels,
    kmeans_predictions, manual_predictions, expected_labels,
    kmeans_accuracy, manual_accuracy, cluster_mapping
):
    """
    Generate the main visualization with 6 subplots.

    Args:
        train_vectors: Training vectors array
        test_vectors: Test vectors array
        kmeans_labels: K-Means cluster labels (integers)
        manual_labels: Manual category labels
        kmeans_predictions: K-Means predictions for test set
        manual_predictions: Manual label predictions for test set
        expected_labels: Expected labels for test set
        kmeans_accuracy: Accuracy of K-Means approach
        manual_accuracy: Accuracy of manual labels approach
        cluster_mapping: Mapping from cluster numbers to Greek letters

    Returns:
        str: Filename where visualization was saved
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. K-Means Clustering Visualization (2D PCA projection)
    ax1 = plt.subplot(2, 3, 1)
    pca = PCA(n_components=2)
    train_vectors_2d = pca.fit_transform(train_vectors)

    # Color map for clusters
    cluster_colors = {0: "red", 1: "blue", 2: "green"}
    colors = [cluster_colors[label] for label in kmeans_labels]

    ax1.scatter(train_vectors_2d[:, 0], train_vectors_2d[:, 1], c=colors, alpha=0.6, s=100)
    ax1.set_title("K-Means Clustering (PCA Projection)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor="red", label="α"),
        Patch(facecolor="blue", label="β"),
        Patch(facecolor="green", label="γ"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # 2. Manual Labels Visualization (2D PCA projection)
    ax2 = plt.subplot(2, 3, 2)
    manual_colors_map = {"A": "gold", "B": "crimson", "C": "steelblue"}
    manual_colors = [manual_colors_map[label] for label in manual_labels]

    ax2.scatter(train_vectors_2d[:, 0], train_vectors_2d[:, 1], c=manual_colors, alpha=0.6, s=100)
    ax2.set_title("Manual Labels (PCA Projection)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    ax2.grid(True, alpha=0.3)

    legend_elements2 = [
        Patch(facecolor="gold", label="A - Hope/Aspiration"),
        Patch(facecolor="crimson", label="B - Conflict/Violence"),
        Patch(facecolor="steelblue", label="C - Science/Technology"),
    ]
    ax2.legend(handles=legend_elements2, loc="upper right")

    # 3. Cluster Distribution
    ax3 = plt.subplot(2, 3, 3)
    kmeans_labels_greek = [cluster_mapping[label] for label in kmeans_labels]
    cluster_counts = Counter(kmeans_labels_greek)
    ax3.bar(cluster_counts.keys(), cluster_counts.values(), color=["red", "blue", "green"])
    ax3.set_title("K-Means Cluster Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Number of Sentences")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Manual Label Distribution
    ax4 = plt.subplot(2, 3, 4)
    manual_counts = Counter(manual_labels)
    ax4.bar(manual_counts.keys(), manual_counts.values(), color=["gold", "crimson", "steelblue"])
    ax4.set_title("Manual Label Distribution", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Category")
    ax4.set_ylabel("Number of Sentences")
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Accuracy Comparison
    ax5 = plt.subplot(2, 3, 5)
    approaches = ["K-Means\nClusters", "Manual\nLabels"]
    accuracies = [kmeans_accuracy * 100, manual_accuracy * 100]
    bars = ax5.bar(approaches, accuracies, color=["lightcoral", "lightgreen"], width=0.6)
    ax5.set_title("k-NN Classification Accuracy Comparison", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Accuracy (%)")
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 6. Confusion Matrix Style Heatmap
    ax6 = plt.subplot(2, 3, 6)
    label_order = ["A", "B", "C"]
    confusion_data = np.zeros((3, 3))

    for i, expected in enumerate(expected_labels):
        predicted = manual_predictions[i]
        exp_idx = label_order.index(expected)
        pred_idx = label_order.index(predicted)
        confusion_data[exp_idx][pred_idx] += 1

    im = ax6.imshow(confusion_data, cmap="Blues", aspect="auto")
    ax6.set_title("Manual Labels: Expected vs Predicted", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Predicted Label")
    ax6.set_ylabel("Expected Label")
    ax6.set_xticks(range(3))
    ax6.set_yticks(range(3))
    ax6.set_xticklabels(label_order)
    ax6.set_yticklabels(label_order)

    for i in range(3):
        for j in range(3):
            ax6.text(j, i, int(confusion_data[i, j]), ha="center", va="center",
                    color="black", fontweight="bold")

    plt.colorbar(im, ax=ax6)

    plt.tight_layout()
    filename = "docs/sentiment_analysis_results.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return filename


def generate_detailed_clustering_analysis(
    train_vectors, test_vectors, kmeans_model, kmeans_labels, manual_labels,
    manual_predictions, expected_labels, cluster_mapping
):
    """
    Generate detailed clustering analysis with 3 subplots.

    Args:
        train_vectors: Training vectors array
        test_vectors: Test vectors array
        kmeans_model: Fitted KMeans object
        kmeans_labels: K-Means cluster labels (integers)
        manual_labels: Manual category labels
        manual_predictions: Manual label predictions for test set
        expected_labels: Expected labels for test set
        cluster_mapping: Mapping from cluster numbers to Greek letters

    Returns:
        tuple: (filename, cluster_analysis_summary)
    """
    fig2 = plt.figure(figsize=(18, 6))

    # PCA transformation
    pca_detailed = PCA(n_components=2, random_state=42)
    train_vectors_2d_detailed = pca_detailed.fit_transform(train_vectors)
    test_vectors_2d = pca_detailed.transform(test_vectors)

    # Graph 1: 2D PCA Visualization of K-Means Clusters
    ax1 = plt.subplot(1, 3, 1)

    cluster_color_map = {0: 'red', 1: 'blue', 2: 'green'}
    cluster_label_map = {0: 'α', 1: 'β', 2: 'γ'}

    for cluster_id in range(3):
        mask = kmeans_labels == cluster_id
        ax1.scatter(
            train_vectors_2d_detailed[mask, 0],
            train_vectors_2d_detailed[mask, 1],
            c=cluster_color_map[cluster_id],
            label=f'Cluster {cluster_label_map[cluster_id]} (n={np.sum(mask)})',
            s=100, alpha=0.7, edgecolors='black', linewidth=0.5
        )

    centers_2d = pca_detailed.transform(kmeans_model.cluster_centers_)
    ax1.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        c='black', marker='X', s=300, edgecolors='yellow', linewidth=2,
        label='Centroids', zorder=5
    )

    ax1.set_title('2D PCA Visualization of K-Means Clusters (K=3)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Principal Component 1', fontsize=11)
    ax1.set_ylabel('Principal Component 2', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Graph 2: Bar Chart of Cluster Size Distribution
    ax2 = plt.subplot(1, 3, 2)

    cluster_sizes = [np.sum(kmeans_labels == i) for i in range(3)]
    cluster_names = ['α', 'β', 'γ']
    colors = ['red', 'blue', 'green']

    bars = ax2.bar(cluster_names, cluster_sizes, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(size)}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax2.set_title('Cluster Size Distribution Post K-Means', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cluster Labels', fontsize=11)
    ax2.set_ylabel('Number of Samples', fontsize=11)
    ax2.set_ylim(0, max(cluster_sizes) * 1.15)
    ax2.grid(True, alpha=0.3, axis='y')

    imbalance_ratio = max(cluster_sizes) / min(cluster_sizes)
    ax2.text(0.5, 0.95, f'Imbalance Ratio: {imbalance_ratio:.1f}:1',
            transform=ax2.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Graph 3: k-NN Analysis - Neighbors of T1
    ax3 = plt.subplot(1, 3, 3)

    manual_color_map = {'A': 'gold', 'B': 'crimson', 'C': 'steelblue'}
    for label in ['A', 'B', 'C']:
        mask = np.array(manual_labels) == label
        ax3.scatter(
            train_vectors_2d_detailed[mask, 0],
            train_vectors_2d_detailed[mask, 1],
            c=manual_color_map[label], label=f'Training: {label}',
            s=80, alpha=0.6, edgecolors='black', linewidth=0.5
        )

    t1_point = test_vectors_2d[0]
    ax3.scatter(t1_point[0], t1_point[1], c='black', marker='*', s=600,
               edgecolors='yellow', linewidth=2, label='T1 (Test Sentence)', zorder=5)

    # Find nearest neighbors
    from sklearn.neighbors import NearestNeighbors
    knn_finder = NearestNeighbors(n_neighbors=5)
    knn_finder.fit(train_vectors)
    distances, indices = knn_finder.kneighbors(test_vectors[0].reshape(1, -1))

    for idx in indices[0]:
        neighbor_point = train_vectors_2d_detailed[idx]
        ax3.plot([t1_point[0], neighbor_point[0]], [t1_point[1], neighbor_point[1]],
                'k--', alpha=0.4, linewidth=1.5, zorder=1)
        ax3.scatter(neighbor_point[0], neighbor_point[1], c='orange', s=200,
                   edgecolors='black', linewidth=2, zorder=4)

    neighbor_labels = [manual_labels[idx] for idx in indices[0]]
    neighbor_counts = {label: neighbor_labels.count(label) for label in ['A', 'B', 'C']}
    annotation_text = f"T1's 5-NN:\n"
    for label in ['A', 'B', 'C']:
        if neighbor_counts[label] > 0:
            annotation_text += f"{label}: {neighbor_counts[label]}  "

    ax3.text(0.02, 0.98, annotation_text.strip(), transform=ax3.transAxes,
            ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            fontweight='bold')

    predicted_label = manual_predictions[0]
    expected_label = expected_labels[0]
    result_color = 'green' if predicted_label == expected_label else 'red'
    ax3.text(0.98, 0.02, f'Predicted: {predicted_label}\nExpected: {expected_label}',
            transform=ax3.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.3),
            fontweight='bold')

    ax3.set_title('k-NN Analysis: Neighbors of T1 (Manual Labels)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Principal Component 1', fontsize=11)
    ax3.set_ylabel('Principal Component 2', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 0.92))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = "docs/detailed_clustering_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    summary = {
        'cluster_sizes': cluster_sizes,
        'imbalance_ratio': imbalance_ratio,
        'neighbor_labels': neighbor_labels,
        'neighbor_counts': neighbor_counts,
        'predicted_label': predicted_label,
        'expected_label': expected_label
    }

    return filename, summary
