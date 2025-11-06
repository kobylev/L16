"""
Main Script - Sentiment Analysis Pipeline (War and Peace Dataset)
Orchestrates the complete K-Means & k-NN sentiment analysis workflow
Author: KobyLev
Dataset: Sentences inspired by themes from "War and Peace" by Leo Tolstoy
"""

import os
from collections import Counter

# Import custom modules
from data_war_and_peace import training_sentences, manual_labels, all_test_sentences, all_expected_labels
from utils import (
    setup_environment, initialize_tokenizer, count_tokens, get_total_tokens,
    calculate_token_cost, CLAUDE_MODEL, reset_token_count
)
from vectorization import vectorize_and_normalize, get_vector_info
from clustering import (
    perform_kmeans_clustering, analyze_cluster_alignment, get_cluster_distribution
)
from classification import (
    train_knn_on_kmeans, train_knn_on_manual_labels,
    predict_with_kmeans_labels, predict_with_manual_labels
)
from analysis import (
    calculate_alignment_accuracy, calculate_test_accuracies,
    generate_results_table, generate_conclusion, print_clustering_summary
)
from visualization import (
    generate_main_visualization, generate_detailed_clustering_analysis
)


def main(num_test_sentences):
    """Main execution function for the sentiment analysis pipeline."""

    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    print("=" * 80)
    print("SENTIMENT ANALYSIS PIPELINE - K-MEANS & k-NN")
    print("Dataset: War and Peace by Leo Tolstoy")
    print(f"Model: {CLAUDE_MODEL}")
    print("=" * 80)
    print()

    # Create docs folder if it doesn't exist
    os.makedirs("docs", exist_ok=True)

    # Setup environment
    api_key = setup_environment()
    initialize_tokenizer()
    reset_token_count()

    # Select test sentences based on parameter
    if num_test_sentences <= len(all_test_sentences):
        test_sentences = all_test_sentences[:num_test_sentences]
        expected_labels = all_expected_labels[:num_test_sentences]
    else:
        test_sentences = all_test_sentences
        expected_labels = all_expected_labels
        print(f"Note: Only {len(all_test_sentences)} test sentences available. Using all.")
        print()

    # ============================================================================
    # STEP 1: DATA SETUP
    # ============================================================================
    print("[STEP 1] Loading Training and Test Datasets")
    print("-" * 80)

    print(f"Training sentences: {len(training_sentences)}")
    print(f"Test sentences: {len(test_sentences)}")
    print(f"Manual label distribution: {Counter(manual_labels)}")
    print()

    # ============================================================================
    # STEP 2: VECTORIZATION AND NORMALIZATION
    # ============================================================================
    print("[STEP 2] Vectorization and Normalization")
    print("-" * 80)

    # Count tokens for all sentences
    all_sentences = training_sentences + test_sentences
    for sentence in all_sentences:
        count_tokens(sentence)

    # Vectorize and normalize
    train_vectors, test_vectors, vectorizer = vectorize_and_normalize(
        training_sentences, test_sentences, max_features=100
    )

    # Print vector info
    vector_info = get_vector_info(train_vectors, test_vectors)
    print(f"Vector dimensions: {vector_info['dimensions']}")
    print(f"Training vectors shape: {vector_info['train_shape']}")
    print(f"Test vectors shape: {vector_info['test_shape']}")
    print(f"Vectors normalized: {vector_info['normalization']} norm")
    print()

    # ============================================================================
    # STEP 3: K-MEANS CLUSTERING (K=3)
    # ============================================================================
    print("[STEP 3] K-Means Clustering (K=3)")
    print("-" * 80)

    # Perform K-Means clustering
    kmeans, kmeans_labels, kmeans_labels_greek = perform_kmeans_clustering(
        train_vectors, n_clusters=3, random_state=42
    )

    print(f"K-Means cluster distribution: {get_cluster_distribution(kmeans_labels_greek)}")
    print()

    # Analyze alignment between manual labels and K-Means clusters
    print("Alignment Analysis:")
    print("-" * 40)

    alignment_result = analyze_cluster_alignment(kmeans_labels, manual_labels)
    cluster_themes = alignment_result['cluster_themes']
    cluster_info = alignment_result['cluster_info']
    cluster_mapping = alignment_result['cluster_mapping']

    for greek, info in cluster_info.items():
        print(f"Cluster {greek}: {info['label_counts']}")
        print(f"  → Dominant manual label: {info['dominant_label']}")
        print(f"  → Theme: {info['theme']}")
        print()

    # Calculate alignment accuracy
    alignment_accuracy = calculate_alignment_accuracy(
        kmeans_labels, manual_labels, cluster_themes
    )
    print(f"K-Means to Manual Label Alignment Accuracy: {alignment_accuracy:.2%}")
    print()

    # ============================================================================
    # STEP 4: k-NN CLASSIFICATION (k=5)
    # ============================================================================
    print("[STEP 4] k-NN Classification (k=5)")
    print("-" * 80)

    # Prediction 1: Using K-Means cluster labels
    print("Prediction 1: k-NN trained on K-Means cluster labels")
    knn_kmeans = train_knn_on_kmeans(train_vectors, kmeans_labels, n_neighbors=5)
    kmeans_predictions, kmeans_predictions_greek = predict_with_kmeans_labels(
        knn_kmeans, test_vectors, cluster_mapping
    )
    print(f"Test predictions (K-Means): {kmeans_predictions_greek}")
    print()

    # Prediction 2: Using Manual labels
    print("Prediction 2: k-NN trained on Manual labels")
    knn_manual = train_knn_on_manual_labels(train_vectors, manual_labels, n_neighbors=5)
    manual_predictions = predict_with_manual_labels(knn_manual, test_vectors)
    print(f"Test predictions (Manual): {list(manual_predictions)}")
    print()

    # ============================================================================
    # STEP 5: RESULTS SUMMARY AND ANALYSIS
    # ============================================================================
    print("[STEP 5] Results Summary")
    print("=" * 80)

    # Create results table
    print("\nTest Set Predictions Comparison:")
    print("-" * 80)
    print(f"{'ID':<5} {'Sentence':<50} {'Expected':<10} {'K-Means':<10} {'Manual':<10}")
    print("-" * 80)

    results = generate_results_table(
        test_sentences, expected_labels, kmeans_predictions_greek, manual_predictions
    )

    for result in results:
        print(f"{result['id']:<5} {result['sentence']:<50} {result['expected']:<10} "
              f"{result['kmeans_pred']:<10} {result['manual_pred']:<10}")

    print("-" * 80)
    print()

    # Calculate accuracy for both approaches
    accuracy_results = calculate_test_accuracies(
        kmeans_predictions, manual_predictions, expected_labels, cluster_themes
    )
    kmeans_accuracy = accuracy_results['kmeans_accuracy']
    manual_accuracy = accuracy_results['manual_accuracy']

    print("Accuracy Comparison:")
    print("-" * 40)
    print(f"k-NN with K-Means labels: {kmeans_accuracy:.2%}")
    print(f"k-NN with Manual labels:  {manual_accuracy:.2%}")
    print()

    # ============================================================================
    # STEP 6: FINAL CONCLUSION
    # ============================================================================
    print("[STEP 6] Final Analysis and Conclusion")
    print("=" * 80)

    conclusion = generate_conclusion(kmeans_accuracy, manual_accuracy)

    for line in conclusion['analysis_lines']:
        print(line)

    print()
    print(f"Recommended approach: {conclusion['better_approach']}")
    print()

    # ============================================================================
    # STEP 7: TOKEN USAGE REPORT
    # ============================================================================
    print(f"[STEP 7] Token Usage Report ({CLAUDE_MODEL})")
    print("=" * 80)

    total_tokens = get_total_tokens()
    cost_info = calculate_token_cost(total_tokens, CLAUDE_MODEL)

    print(f"Total tokens processed: {cost_info['total_tokens']:,}")
    print(f"Average tokens per sentence: {total_tokens / len(all_sentences):.2f}")
    print()
    print(f"Model: {cost_info['model']}")
    print(f"Estimated cost (input tokens only): ${cost_info['input_cost']:.6f}")
    print()

    # ============================================================================
    # STEP 8: VISUALIZATION
    # ============================================================================
    print("[STEP 8] Generating Visualization Graphs")
    print("=" * 80)

    # Save with second_ prefix
    main_viz_file = f"docs/second_sentiment_analysis_results_{num_test_sentences}.png"

    # Generate visualizations (we'll need to modify visualization.py temporarily)
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from matplotlib.patches import Patch
    import numpy as np

    # Quick inline visualization to save with custom name
    fig = plt.figure(figsize=(16, 10))

    # PCA projection
    ax1 = plt.subplot(2, 3, 1)
    pca = PCA(n_components=2)
    train_vectors_2d = pca.fit_transform(train_vectors)
    cluster_colors = {0: "red", 1: "blue", 2: "green"}
    colors = [cluster_colors[label] for label in kmeans_labels]
    ax1.scatter(train_vectors_2d[:, 0], train_vectors_2d[:, 1], c=colors, alpha=0.6, s=100)
    ax1.set_title("K-Means Clustering (PCA)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(main_viz_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Visualization saved as '{main_viz_file}'")
    print()

    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    return {
        'kmeans_accuracy': kmeans_accuracy,
        'manual_accuracy': manual_accuracy,
        'alignment_accuracy': alignment_accuracy,
        'cluster_distribution': dict(get_cluster_distribution(kmeans_labels_greek)),
        'total_tokens': total_tokens,
        'cost': cost_info['input_cost']
    }


if __name__ == "__main__":
    import sys
    num_test = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(num_test)
