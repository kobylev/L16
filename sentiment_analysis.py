"""
Combined Sentiment Analysis using K-Means Clustering and k-NN Classification
Author: KobyLev
Dataset: Sentences inspired by themes from "The Sparrow" by Mary Doria Russell
"""

import os
import sys
import numpy as np
import tiktoken
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from collections import Counter

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Claude Haiku Model Configuration
CLAUDE_MODEL = "claude-3-haiku-20240307"  # Most cost-effective Claude model
# Note: This script uses local TF-IDF vectorization for efficiency
# If you need Claude API embeddings, the model above would be used

# Initialize token counter for Claude Haiku
# Claude uses a different tokenizer, but we can approximate with cl100k_base
# which is closer to Claude's tokenization than GPT models
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Closest to Claude's tokenizer
except:
    tokenizer = tiktoken.encoding_for_model("gpt-4")  # Fallback

total_tokens = 0


def count_tokens(text):
    """Count tokens in text using tiktoken (approximation for Claude Haiku)."""
    global total_tokens
    tokens = len(tokenizer.encode(str(text)))
    total_tokens += tokens
    return tokens


print("=" * 80)
print("SENTIMENT ANALYSIS PIPELINE - K-MEANS & k-NN")
print(f"Model: {CLAUDE_MODEL}")
print("=" * 80)
print()

# Prompt user for number of test sentences
print("Configuration:")
print("-" * 80)
user_input = input(
    "Enter number of test sentences (press Enter for default=100): "
).strip()

if user_input == "":
    num_test_sentences = 100
    print(f"Using default: {num_test_sentences} test sentences")
else:
    try:
        num_test_sentences = int(user_input)
        if num_test_sentences < 1:
            print("Invalid input. Using default: 100 test sentences")
            num_test_sentences = 100
        else:
            print(f"Using: {num_test_sentences} test sentences")
    except ValueError:
        print("Invalid input. Using default: 100 test sentences")
        num_test_sentences = 100

print()
print("=" * 80)
print()

# ============================================================================
# STEP 1: DATA SETUP
# ============================================================================
print("[STEP 1] Loading Training and Test Datasets")
print("-" * 80)

# Training Dataset (30 sentences inspired by "The Sparrow" themes)
training_sentences = [
    "The linguist decoded patterns in the alien transmission.",
    "They hoped this mission would reveal God's greater purpose.",
    "The attack came without warning in the darkness.",
    "Father Emilio believed faith would guide their journey.",
    "The spacecraft's hull was designed for interstellar travel.",
    "Violence erupted as territorial boundaries were violated.",
    "Sofia dreamed of making first contact with another species.",
    "The airlock mechanism required precise engineering calculations.",
    "Suffering tested the limits of their religious conviction.",
    "The Jesuit mission sought knowledge beyond the stars.",
    "Navigation systems plotted coordinates through deep space.",
    "The massacre left survivors traumatized and broken.",
    "Revenge consumed those who had lost everything.",
    "The probe's sensors detected complex signal patterns.",
    "Anne yearned for understanding across the cosmic divide.",
    "The telescope array captured stunning celestial imagery.",
    "Betrayal shattered the trust between species forever.",
    "Jimmy anticipated breakthrough discoveries on the alien world.",
    "Battle cries echoed through the alien settlement.",
    "The habitat modules provided life support systems.",
    "Emilio planned redemption through continued service.",
    "The observatory stood as humanity's window to infinity.",
    "Conflict destroyed any chance of peaceful coexistence.",
    "The young priest studied xenolinguistic communication protocols.",
    "Her determination drove scientific progress forward relentlessly.",
    "The ambush cost them their most skilled navigator.",
    "They aspired to bridge the gap between worlds.",
    "The ship's framework incorporated revolutionary composite materials.",
    "The raid left the expedition decimated and stranded.",
    "She dreamed of understanding alien consciousness itself.",
]

manual_labels = [
    "C",
    "A",
    "B",
    "A",
    "C",
    "B",
    "A",
    "C",
    "A",
    "A",
    "C",
    "B",
    "B",
    "C",
    "A",
    "C",
    "B",
    "A",
    "B",
    "C",
    "A",
    "C",
    "B",
    "C",
    "A",
    "B",
    "A",
    "C",
    "B",
    "A",
]

# Extended Test Dataset (expandable pool - inspired by "The Sparrow")
all_test_sentences = [
    # Category A: Hope/Aspiration (40 sentences)
    "The mission promised unprecedented scientific discoveries.",
    "He vowed to find meaning in the cosmic journey.",
    "The crew celebrated receiving signals from Rakhat.",
    "She hoped to establish peaceful interspecies communication.",
    "They dreamed of understanding alien musical traditions.",
    "The scientists anticipated revolutionary linguistic breakthroughs.",
    "Emilio aspired to serve God among the stars.",
    "The expedition team looked forward to landfall.",
    "Anne wished for her research to bridge two worlds.",
    "Father Sandoz envisioned a new era of faith.",
    "The Jesuits hoped their mission would succeed.",
    "Sofia planned to document the first contact.",
    "The priest sought redemption through continued devotion.",
    "The musicians dreamed of sharing beauty across species.",
    "Jimmy longed to see alien civilization firsthand.",
    "The crew prayed for safe passage through space.",
    "She aspired to decode the alien language completely.",
    "The linguists yearned to master xenocommunication.",
    "They hoped the aliens would welcome their arrival.",
    "The Father wished for divine guidance on Rakhat.",
    "Emilio dreamed of finding God's purpose revealed.",
    "The team anticipated mutual understanding and peace.",
    "Anne vowed to protect indigenous culture from harm.",
    "The missionaries prayed for wisdom and compassion.",
    "Sandoz hoped to unite humanity with alien faith.",
    "They longed for meaningful contact with the Jana'ata.",
    "The young priest dreamed of spiritual enlightenment.",
    "Sofia planned to establish cultural exchange protocols.",
    "The crew hoped to return home with knowledge.",
    "She wished for humanity to learn from aliens.",
    "The astronomers looked forward to observing new stars.",
    "Jimmy aspired to compose music with alien harmonies.",
    "The expedition hoped for breakthrough discoveries daily.",
    "Father Emilio envisioned a thriving interstellar mission.",
    "They dreamed of escaping Earth's limitations forever.",
    "The researchers hoped their work would endure.",
    "Anne planned a comprehensive xenoanthropological study.",
    "The crew wished for abundant resources on Rakhat.",
    "The Jesuit hoped to prove faith transcends worlds.",
    "They aspired to build lasting bonds between species.",
    # Category B: Conflict/Violence (40 sentences)
    "The brutal interrogation left Emilio physically broken.",
    "Blood marked the ground where the massacre occurred.",
    "The Jana'ata warriors attacked with overwhelming force.",
    "Alien soldiers captured the defenseless expedition members.",
    "Weapons clashed as territorial disputes exploded violently.",
    "The ambush came without warning in darkness.",
    "Flames consumed the settlement as refugees fled screaming.",
    "The executioner prepared the captive for punishment.",
    "Raiders destroyed the camp and killed the crew.",
    "The torture chamber echoed with Sandoz's screams.",
    "Projectiles rained down upon the fleeing humans.",
    "The alien forces slaughtered everyone in their path.",
    "War destroyed everything the mission accomplished.",
    "The assassin struck Askama swiftly at night.",
    "Blood soaked the alien soil after battle.",
    "Enemies burned supplies to starve the survivors.",
    "The mob attacked with primitive brutal weapons.",
    "Soldiers dragged prisoners to holding cells.",
    "The rebellion was crushed with savage efficiency.",
    "Rioters destroyed scientific equipment and resources.",
    "The guards prepared captives for execution rituals.",
    "Mercenaries killed without hesitation or conscience.",
    "The siege left the expedition decimated and helpless.",
    "Warriors charged with deadly intent and rage.",
    "The compound fell after a devastating assault.",
    "Betrayal led to murder and violent retribution.",
    "The confrontation ended with fatal injuries inflicted.",
    "Bandits ambushed the supply expedition mercilessly.",
    "The raid left nothing but ruins and bodies.",
    "Combatants fought savagely with alien weaponry.",
    "The invasion brought death and total destruction.",
    "Assassins poisoned food supplies systematically.",
    "The battle raged for days with horrific casualties.",
    "Soldiers burned shelters with people trapped inside.",
    "The massacre spared no one from violence.",
    "Warriors trampled victims beneath armored feet.",
    "The conflict escalated into genocidal warfare.",
    "Rebels attacked the settlement under cover of darkness.",
    "The conquerors showed no mercy to survivors.",
    "Violence erupted suddenly in the crowded alien city.",
    # Category C: Science/Technology (40 sentences)
    "The mission was dedicated to advancing scientific knowledge.",
    "They carefully installed the complex sensor arrays.",
    "The engineers measured every trajectory precisely.",
    "The chief technician inspected each system carefully.",
    "Antenna arrays rose majestically toward the stars.",
    "The spacecraft's foundation was engineered deep and strong.",
    "Technicians programmed intricate navigation patterns.",
    "The support struts reinforced the ship's hull.",
    "Workers assembled modular habitat components systematically.",
    "The vessel's main corridor stretched impressively long.",
    "Engineers calibrated instruments for atmospheric analysis.",
    "The docking bay intersected the corridor at right angles.",
    "Equipment racks climbed to maximize storage space.",
    "The observatory curved gracefully behind the command deck.",
    "Fuel was mixed with precise chemical proportions.",
    "The storage bay was constructed beneath the main deck.",
    "Solar panels were deployed in overlapping configurations.",
    "The communication tower transmitted farther than any predecessor.",
    "Structural beams distributed stress efficiently throughout.",
    "Viewports were positioned to maximize stellar observation.",
    "The asteroid provided high-quality mineral resources.",
    "Technicians constructed frameworks for the laboratory modules.",
    "The ventilation system circulated oxygen throughout the ship.",
    "Equipment was calibrated and tested at the facility.",
    "The corridor circled around the central core.",
    "Life support required redundant backup systems.",
    "The laboratory module adjoined the main living quarters.",
    "Composite materials supported the pressure-bearing structure.",
    "The airlock featured sophisticated decontamination systems.",
    "Calculations were verified repeatedly for accuracy.",
    "The workstations were positioned for optimal workflow.",
    "Shields reinforced the hull against radiation strategically.",
    "The main viewport dominated the observation deck.",
    "Systems were spaced according to engineering specifications.",
    "The corridor provided protected pathways for crew.",
    "Structural members converged at the central axis.",
    "The antenna array pointed precisely toward Earth.",
    "Access tubes spiraled throughout the ship's interior.",
    "The deck was constructed with anti-slip composite materials.",
    "Assembly followed the engineer's detailed schematics.",
]

all_expected_labels = (
    ["A"] * 40  # Hope/Aspiration
    + ["B"] * 40  # Conflict/Violence
    + ["C"] * 40  # Architecture/Building
)

# Select test sentences based on user input
if num_test_sentences <= len(all_test_sentences):
    test_sentences = all_test_sentences[:num_test_sentences]
    expected_labels = all_expected_labels[:num_test_sentences]
else:
    # If user requests more than available, use all and inform
    test_sentences = all_test_sentences
    expected_labels = all_expected_labels
    print(f"Note: Only {len(all_test_sentences)} test sentences available. Using all.")
    print()

print(f"Training sentences: {len(training_sentences)}")
print(f"Test sentences: {len(test_sentences)}")
print(f"Manual label distribution: {Counter(manual_labels)}")
print()

# ============================================================================
# STEP 2: VECTORIZATION AND NORMALIZATION
# ============================================================================
print("[STEP 2] Vectorization and Normalization")
print("-" * 80)

# Combine all sentences for consistent vectorization
all_sentences = training_sentences + test_sentences

# Count tokens for all sentences
for sentence in all_sentences:
    count_tokens(sentence)

# Use TfidfVectorizer for efficient vectorization
vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
all_vectors = vectorizer.fit_transform(all_sentences).toarray()

# L2 Normalization
all_vectors_normalized = normalize(all_vectors, norm="l2")

# Split back into training and test sets
train_vectors = all_vectors_normalized[: len(training_sentences)]
test_vectors = all_vectors_normalized[len(training_sentences) :]

print(f"Vector dimensions: {train_vectors.shape[1]}")
print(f"Training vectors shape: {train_vectors.shape}")
print(f"Test vectors shape: {test_vectors.shape}")
print(f"Vectors normalized: L2 norm")
print()

# ============================================================================
# STEP 3: K-MEANS CLUSTERING (K=3)
# ============================================================================
print("[STEP 3] K-Means Clustering (K=3)")
print("-" * 80)

# Run K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(train_vectors)

# Map cluster numbers to Greek letters for clarity
cluster_mapping = {0: "α", 1: "β", 2: "γ"}
kmeans_labels_greek = [cluster_mapping[label] for label in kmeans_labels]

print(f"K-Means cluster distribution: {Counter(kmeans_labels_greek)}")
print()

# Analyze alignment between manual labels and K-Means clusters
print("Alignment Analysis:")
print("-" * 40)

# Create a mapping to find most common manual label per cluster
cluster_analysis = {0: [], 1: [], 2: []}
for i, cluster in enumerate(kmeans_labels):
    cluster_analysis[cluster].append(manual_labels[i])

# Find dominant theme in each cluster
cluster_themes = {}
for cluster_num, labels in cluster_analysis.items():
    label_counts = Counter(labels)
    dominant_label = label_counts.most_common(1)[0][0]
    cluster_themes[cluster_num] = dominant_label

    greek = cluster_mapping[cluster_num]
    print(f"Cluster {greek}: {dict(label_counts)}")
    print(f"  → Dominant manual label: {dominant_label}")

    # Determine theme
    if dominant_label == "A":
        theme = "Hope/Aspiration"
    elif dominant_label == "B":
        theme = "Conflict/Violence"
    else:
        theme = "Science/Technology"
    print(f"  → Theme: {theme}")
    print()

# Calculate alignment accuracy
# Best mapping from K-Means to manual labels
aligned_predictions = [cluster_themes[label] for label in kmeans_labels]
alignment_accuracy = accuracy_score(manual_labels, aligned_predictions)

print(f"K-Means to Manual Label Alignment Accuracy: {alignment_accuracy:.2%}")
print()

# ============================================================================
# STEP 4: k-NN CLASSIFICATION (k=5)
# ============================================================================
print("[STEP 4] k-NN Classification (k=5)")
print("-" * 80)

# Prediction 1: Using K-Means cluster labels (α, β, γ)
print("Prediction 1: k-NN trained on K-Means cluster labels")
knn_kmeans = KNeighborsClassifier(n_neighbors=5)
knn_kmeans.fit(train_vectors, kmeans_labels)
kmeans_predictions = knn_kmeans.predict(test_vectors)
kmeans_predictions_greek = [cluster_mapping[label] for label in kmeans_predictions]

print(f"Test predictions (K-Means): {kmeans_predictions_greek}")
print()

# Prediction 2: Using Manual labels (A, B, C)
print("Prediction 2: k-NN trained on Manual labels")
knn_manual = KNeighborsClassifier(n_neighbors=5)
knn_manual.fit(train_vectors, manual_labels)
manual_predictions = knn_manual.predict(test_vectors)

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

for i, sentence in enumerate(test_sentences):
    test_id = f"T{i+1}"
    short_sentence = sentence[:47] + "..." if len(sentence) > 50 else sentence
    expected = expected_labels[i]
    kmeans_pred = kmeans_predictions_greek[i]
    manual_pred = manual_predictions[i]

    print(
        f"{test_id:<5} {short_sentence:<50} {expected:<10} {kmeans_pred:<10} {manual_pred:<10}"
    )

print("-" * 80)
print()

# Calculate accuracy for both approaches
kmeans_mapped_predictions = [cluster_themes[pred] for pred in kmeans_predictions]
kmeans_accuracy = accuracy_score(expected_labels, kmeans_mapped_predictions)
manual_accuracy = accuracy_score(expected_labels, manual_predictions)

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

if manual_accuracy > kmeans_accuracy:
    better_approach = "Manual labels (A, B, C)"
    difference = manual_accuracy - kmeans_accuracy
    print(f"✓ The Manual labels resulted in BETTER classification performance.")
    print(f"  Accuracy difference: +{difference:.2%}")
    print()
    print("Analysis:")
    print("  - Manual labels provide supervised categorization based on semantic")
    print("    understanding of Hope/Aspiration, Conflict/Violence, and Science/Technology.")
    print("  - K-Means clustering found patterns but may not perfectly align with")
    print("    human-defined semantic categories.")
elif kmeans_accuracy > manual_accuracy:
    better_approach = "K-Means labels (α, β, γ)"
    difference = kmeans_accuracy - manual_accuracy
    print(f"✓ The K-Means labels resulted in BETTER classification performance.")
    print(f"  Accuracy difference: +{difference:.2%}")
    print()
    print("Analysis:")
    print("  - K-Means discovered natural groupings in the vectorized space that")
    print("    better represent the underlying patterns in the test data.")
    print("  - This suggests the unsupervised clusters may capture nuances not")
    print("    reflected in the manual categorical labels.")
else:
    better_approach = "Both approaches (tied)"
    print(f"✓ Both approaches achieved EQUAL classification performance.")
    print(f"  Accuracy: {manual_accuracy:.2%}")
    print()
    print("Analysis:")
    print("  - Both supervised (manual) and unsupervised (K-Means) approaches")
    print("    yielded identical results on the test set.")
    print("  - The K-Means clusters aligned well with the manual categorization,")
    print("    suggesting coherent semantic groupings in the data.")

print()
print(f"Recommended approach: {better_approach}")
print()

# ============================================================================
# STEP 7: TOKEN USAGE REPORT
# ============================================================================
print(f"[STEP 7] Token Usage Report ({CLAUDE_MODEL})")
print("=" * 80)
print(f"Total tokens processed: {total_tokens:,}")
print(f"Average tokens per sentence: {total_tokens / len(all_sentences):.2f}")
print(f"Training set tokens: ~{sum(count_tokens(s) for s in training_sentences):,}")
print(f"Test set tokens: ~{sum(count_tokens(s) for s in test_sentences):,}")
print()
print(f"Model: {CLAUDE_MODEL}")
print("Note: Token count uses cl100k_base encoding as an approximation.")
print("      Actual Claude API usage may vary slightly.")
print()

# Estimate API cost for Claude Haiku (as of 2024)
# Claude Haiku pricing: $0.25 per million input tokens, $1.25 per million output tokens
input_cost = (total_tokens / 1_000_000) * 0.25
print(f"Estimated cost (input tokens only): ${input_cost:.6f}")
print("(Based on Claude Haiku pricing: $0.25/million input tokens)")
print()

# ============================================================================
# STEP 8: VISUALIZATION
# ============================================================================
print("[STEP 8] Generating Visualization Graphs")
print("=" * 80)

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

# Add legend
from matplotlib.patches import Patch

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

ax2.scatter(
    train_vectors_2d[:, 0], train_vectors_2d[:, 1], c=manual_colors, alpha=0.6, s=100
)
ax2.set_title("Manual Labels (PCA Projection)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.grid(True, alpha=0.3)

# Add legend
legend_elements2 = [
    Patch(facecolor="gold", label="A - Hope/Aspiration"),
    Patch(facecolor="crimson", label="B - Conflict/Violence"),
    Patch(facecolor="steelblue", label="C - Science/Technology"),
]
ax2.legend(handles=legend_elements2, loc="upper right")

# 3. Cluster Distribution
ax3 = plt.subplot(2, 3, 3)
cluster_counts = Counter(kmeans_labels_greek)
ax3.bar(cluster_counts.keys(), cluster_counts.values(), color=["red", "blue", "green"])
ax3.set_title("K-Means Cluster Distribution", fontsize=12, fontweight="bold")
ax3.set_xlabel("Cluster")
ax3.set_ylabel("Number of Sentences")
ax3.grid(True, alpha=0.3, axis="y")

# 4. Manual Label Distribution
ax4 = plt.subplot(2, 3, 4)
manual_counts = Counter(manual_labels)
ax4.bar(
    manual_counts.keys(), manual_counts.values(), color=["gold", "crimson", "steelblue"]
)
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

# Add value labels on bars
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

# 6. Confusion Matrix Style Heatmap - Test Predictions
ax6 = plt.subplot(2, 3, 6)
# Count predictions vs expected for test set
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

# Add text annotations
for i in range(3):
    for j in range(3):
        text = ax6.text(
            j,
            i,
            int(confusion_data[i, j]),
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
        )

plt.colorbar(im, ax=ax6)

plt.tight_layout()
plt.savefig("sentiment_analysis_results.png", dpi=300, bbox_inches="tight")
print("✓ Visualization saved as 'sentiment_analysis_results.png'")
print()

# ============================================================================
# STEP 9: DETAILED CLUSTERING ANALYSIS GRAPHS
# ============================================================================
print("[STEP 9] Generating Detailed Clustering Analysis")
print("=" * 80)

# Create a new figure for the three detailed analysis graphs
fig2 = plt.figure(figsize=(18, 6))

# -------------------------------------------------------------------------
# Graph 1: 2D PCA Visualization of K-Means Clusters
# -------------------------------------------------------------------------
ax1 = plt.subplot(1, 3, 1)

# Use the same PCA model from earlier (already fitted)
pca_detailed = PCA(n_components=2, random_state=42)
train_vectors_2d_detailed = pca_detailed.fit_transform(train_vectors)

# Color mapping for K-Means clusters
cluster_color_map = {0: 'red', 1: 'blue', 2: 'green'}
cluster_label_map = {0: 'α', 1: 'β', 2: 'γ'}

# Plot each cluster with different colors
for cluster_id in range(3):
    mask = kmeans_labels == cluster_id
    ax1.scatter(
        train_vectors_2d_detailed[mask, 0],
        train_vectors_2d_detailed[mask, 1],
        c=cluster_color_map[cluster_id],
        label=f'Cluster {cluster_label_map[cluster_id]} (n={np.sum(mask)})',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

# Plot cluster centers
centers_2d = pca_detailed.transform(kmeans.cluster_centers_)
ax1.scatter(
    centers_2d[:, 0],
    centers_2d[:, 1],
    c='black',
    marker='X',
    s=300,
    edgecolors='yellow',
    linewidth=2,
    label='Centroids',
    zorder=5
)

ax1.set_title('2D PCA Visualization of K-Means Clusters (K=3)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Principal Component 1', fontsize=11)
ax1.set_ylabel('Principal Component 2', fontsize=11)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# Graph 2: Bar Chart of Cluster Size Distribution
# -------------------------------------------------------------------------
ax2 = plt.subplot(1, 3, 2)

cluster_sizes = [np.sum(kmeans_labels == i) for i in range(3)]
cluster_names = ['α', 'β', 'γ']
colors = ['red', 'blue', 'green']

bars = ax2.bar(cluster_names, cluster_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, size in zip(bars, cluster_sizes):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f'{int(size)}',
        ha='center',
        va='bottom',
        fontsize=14,
        fontweight='bold'
    )

ax2.set_title('Cluster Size Distribution Post K-Means', fontsize=14, fontweight='bold')
ax2.set_xlabel('Cluster Labels', fontsize=11)
ax2.set_ylabel('Number of Samples', fontsize=11)
ax2.set_ylim(0, max(cluster_sizes) * 1.15)
ax2.grid(True, alpha=0.3, axis='y')

# Add imbalance warning annotation
imbalance_ratio = max(cluster_sizes) / min(cluster_sizes)
ax2.text(
    0.5, 0.95,
    f'Imbalance Ratio: {imbalance_ratio:.1f}:1',
    transform=ax2.transAxes,
    ha='center',
    va='top',
    fontsize=10,
    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
)

# -------------------------------------------------------------------------
# Graph 3: k-NN Analysis - Neighbors of T1 (Manual Labels)
# -------------------------------------------------------------------------
ax3 = plt.subplot(1, 3, 3)

# Transform test vectors using the same PCA
test_vectors_2d = pca_detailed.transform(test_vectors)

# Plot training points colored by manual labels
manual_color_map = {'A': 'gold', 'B': 'crimson', 'C': 'steelblue'}
for label in ['A', 'B', 'C']:
    mask = np.array(manual_labels) == label
    ax3.scatter(
        train_vectors_2d_detailed[mask, 0],
        train_vectors_2d_detailed[mask, 1],
        c=manual_color_map[label],
        label=f'Training: {label}',
        s=80,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

# Highlight T1 (first test sentence)
t1_point = test_vectors_2d[0]
ax3.scatter(
    t1_point[0],
    t1_point[1],
    c='black',
    marker='*',
    s=600,
    edgecolors='yellow',
    linewidth=2,
    label='T1 (Test Sentence)',
    zorder=5
)

# Find and highlight the 5 nearest neighbors of T1
from sklearn.neighbors import NearestNeighbors
knn_finder = NearestNeighbors(n_neighbors=5)
knn_finder.fit(train_vectors)
distances, indices = knn_finder.kneighbors(test_vectors[0].reshape(1, -1))

# Draw lines from T1 to its 5 nearest neighbors
for i, idx in enumerate(indices[0]):
    neighbor_point = train_vectors_2d_detailed[idx]
    ax3.plot(
        [t1_point[0], neighbor_point[0]],
        [t1_point[1], neighbor_point[1]],
        'k--',
        alpha=0.4,
        linewidth=1.5,
        zorder=1
    )
    # Highlight the neighbor
    ax3.scatter(
        neighbor_point[0],
        neighbor_point[1],
        c='orange',
        s=200,
        edgecolors='black',
        linewidth=2,
        zorder=4
    )

# Add annotation showing neighbor labels
neighbor_labels = [manual_labels[idx] for idx in indices[0]]
neighbor_counts = {label: neighbor_labels.count(label) for label in ['A', 'B', 'C']}
annotation_text = f"T1's 5-NN:\n"
for label in ['A', 'B', 'C']:
    if neighbor_counts[label] > 0:
        annotation_text += f"{label}: {neighbor_counts[label]}  "

ax3.text(
    0.02, 0.98,
    annotation_text.strip(),
    transform=ax3.transAxes,
    ha='left',
    va='top',
    fontsize=10,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
    fontweight='bold'
)

# Add prediction result
predicted_label = manual_predictions[0]
expected_label = expected_labels[0]
result_color = 'green' if predicted_label == expected_label else 'red'
ax3.text(
    0.98, 0.02,
    f'Predicted: {predicted_label}\nExpected: {expected_label}',
    transform=ax3.transAxes,
    ha='right',
    va='bottom',
    fontsize=10,
    bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.3),
    fontweight='bold'
)

ax3.set_title('k-NN Analysis: Neighbors of T1 (Manual Labels)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Principal Component 1', fontsize=11)
ax3.set_ylabel('Principal Component 2', fontsize=11)
ax3.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 0.92))
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("detailed_clustering_analysis.png", dpi=300, bbox_inches="tight")
print("✓ Detailed analysis saved as 'detailed_clustering_analysis.png'")
print()

# Print detailed analysis summary
print("Clustering Analysis Summary:")
print("-" * 80)
print(f"1. K-Means Imbalance: Cluster α dominates with {cluster_sizes[0]} samples")
print(f"   while clusters β and γ have only {cluster_sizes[1]} and {cluster_sizes[2]} samples.")
print()
print(f"2. Imbalance Ratio: {imbalance_ratio:.1f}:1 (largest to smallest cluster)")
print()
print(f"3. T1 Classification Analysis:")
print(f"   - Expected Label: {expected_labels[0]}")
print(f"   - Predicted Label: {manual_predictions[0]}")
print(f"   - T1's 5 Nearest Neighbors: {neighbor_labels}")
print(f"   - Neighbor Vote Count: {neighbor_counts}")
if predicted_label != expected_label:
    winner = max(neighbor_counts, key=neighbor_counts.get)
    print(f"   - Majority vote: '{winner}' caused misclassification")
else:
    print(f"   - Correctly classified by majority vote")
print()

print("=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
