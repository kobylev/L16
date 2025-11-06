"""
Analysis Module - Results and Metrics
Handles accuracy calculation, comparison, and result reporting
"""

from sklearn.metrics import accuracy_score
from collections import Counter


def calculate_alignment_accuracy(kmeans_labels, manual_labels, cluster_themes):
    """
    Calculate alignment accuracy between K-Means clusters and manual labels.

    Args:
        kmeans_labels: K-Means cluster labels (integers)
        manual_labels: Manual category labels
        cluster_themes: Mapping from cluster numbers to dominant labels

    Returns:
        float: Alignment accuracy
    """
    aligned_predictions = [cluster_themes[label] for label in kmeans_labels]
    alignment_accuracy = accuracy_score(manual_labels, aligned_predictions)
    return alignment_accuracy


def calculate_test_accuracies(kmeans_predictions, manual_predictions, expected_labels, cluster_themes):
    """
    Calculate accuracy for both K-Means and manual label approaches.

    Args:
        kmeans_predictions: K-Means predictions (integers)
        manual_predictions: Manual label predictions
        expected_labels: Expected labels for test set
        cluster_themes: Mapping from cluster numbers to dominant labels

    Returns:
        dict: Contains kmeans_accuracy and manual_accuracy
    """
    kmeans_mapped_predictions = [cluster_themes[pred] for pred in kmeans_predictions]
    kmeans_accuracy = accuracy_score(expected_labels, kmeans_mapped_predictions)
    manual_accuracy = accuracy_score(expected_labels, manual_predictions)

    return {
        'kmeans_accuracy': kmeans_accuracy,
        'manual_accuracy': manual_accuracy,
        'kmeans_mapped_predictions': kmeans_mapped_predictions
    }


def generate_results_table(test_sentences, expected_labels, kmeans_predictions_greek, manual_predictions):
    """
    Generate results comparison table data.

    Args:
        test_sentences: List of test sentences
        expected_labels: Expected labels for test set
        kmeans_predictions_greek: K-Means predictions as Greek letters
        manual_predictions: Manual label predictions

    Returns:
        list: List of result dictionaries
    """
    results = []
    for i, sentence in enumerate(test_sentences):
        test_id = f"T{i+1}"
        short_sentence = sentence[:47] + "..." if len(sentence) > 50 else sentence
        expected = expected_labels[i]
        kmeans_pred = kmeans_predictions_greek[i]
        manual_pred = manual_predictions[i]

        results.append({
            'id': test_id,
            'sentence': short_sentence,
            'expected': expected,
            'kmeans_pred': kmeans_pred,
            'manual_pred': manual_pred
        })

    return results


def generate_conclusion(kmeans_accuracy, manual_accuracy):
    """
    Generate analysis conclusion based on accuracy comparison.

    Args:
        kmeans_accuracy: Accuracy of K-Means approach
        manual_accuracy: Accuracy of manual labels approach

    Returns:
        dict: Contains conclusion text and recommendation
    """
    if manual_accuracy > kmeans_accuracy:
        better_approach = "Manual labels (A, B, C)"
        difference = manual_accuracy - kmeans_accuracy
        analysis_lines = [
            f"✓ The Manual labels resulted in BETTER classification performance.",
            f"  Accuracy difference: +{difference:.2%}",
            "",
            "Analysis:",
            "  - Manual labels provide supervised categorization based on semantic",
            "    understanding of Hope/Aspiration, Conflict/Violence, and Science/Technology.",
            "  - K-Means clustering found patterns but may not perfectly align with",
            "    human-defined semantic categories."
        ]
    elif kmeans_accuracy > manual_accuracy:
        better_approach = "K-Means labels (α, β, γ)"
        difference = kmeans_accuracy - manual_accuracy
        analysis_lines = [
            f"✓ The K-Means labels resulted in BETTER classification performance.",
            f"  Accuracy difference: +{difference:.2%}",
            "",
            "Analysis:",
            "  - K-Means discovered natural groupings in the vectorized space that",
            "    better represent the underlying patterns in the test data.",
            "  - This suggests the unsupervised clusters may capture nuances not",
            "    reflected in the manual categorical labels."
        ]
    else:
        better_approach = "Both approaches (tied)"
        analysis_lines = [
            f"✓ Both approaches achieved EQUAL classification performance.",
            f"  Accuracy: {manual_accuracy:.2%}",
            "",
            "Analysis:",
            "  - Both supervised (manual) and unsupervised (K-Means) approaches",
            "    yielded identical results on the test set.",
            "  - The K-Means clusters aligned well with the manual categorization,",
            "    suggesting coherent semantic groupings in the data."
        ]

    return {
        'better_approach': better_approach,
        'analysis_lines': analysis_lines
    }


def print_clustering_summary(cluster_sizes, imbalance_ratio, neighbor_labels,
                            neighbor_counts, predicted_label, expected_label):
    """
    Print clustering analysis summary.

    Args:
        cluster_sizes: List of cluster sizes
        imbalance_ratio: Imbalance ratio between largest and smallest cluster
        neighbor_labels: Labels of T1's nearest neighbors
        neighbor_counts: Count of each label among neighbors
        predicted_label: Predicted label for T1
        expected_label: Expected label for T1
    """
    print("Clustering Analysis Summary:")
    print("-" * 80)
    print(f"1. K-Means Imbalance: Cluster α dominates with {cluster_sizes[0]} samples")
    print(f"   while clusters β and γ have only {cluster_sizes[1]} and {cluster_sizes[2]} samples.")
    print()
    print(f"2. Imbalance Ratio: {imbalance_ratio:.1f}:1 (largest to smallest cluster)")
    print()
    print(f"3. T1 Classification Analysis:")
    print(f"   - Expected Label: {expected_label}")
    print(f"   - Predicted Label: {predicted_label}")
    print(f"   - T1's 5 Nearest Neighbors: {neighbor_labels}")
    print(f"   - Neighbor Vote Count: {neighbor_counts}")
    if predicted_label != expected_label:
        winner = max(neighbor_counts, key=neighbor_counts.get)
        print(f"   - Majority vote: '{winner}' caused misclassification")
    else:
        print(f"   - Correctly classified by majority vote")
    print()
