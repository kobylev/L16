# Sentiment Analysis Pipeline - K-Means & k-NN

## Overview
This project implements a comprehensive sentiment analysis pipeline to answer a fundamental machine learning question: **Can unsupervised clustering match the performance of supervised learning for text classification?**

By combining K-Means clustering (unsupervised) and k-NN classification (supervised), this pipeline compares two distinct approaches to categorizing text into three semantic groups. The dataset consists of sentences inspired by themes from Mary Doria Russell's science fiction novel "The Sparrow", classified into:
- **Hope/Aspiration** - Spiritual journeys, dreams, and discoveries
- **Conflict/Violence** - Battles, suffering, and warfare
- **Science/Technology** - Engineering, spacecraft systems, and technical details

### The Research Question
**Does K-Means clustering discover meaningful semantic patterns that rival human-labeled categories for k-NN classification?**

This project provides a hands-on exploration of:
- The trade-off between unsupervised pattern discovery vs supervised learning
- The limitations of TF-IDF vectorization for semantic understanding
- The impact of cluster imbalance on classification performance
- Cost-effective text analysis using approximated token counting

## Key Findings

![Sentiment Analysis Results](docs/sentiment_analysis_results.png)

### 🏆 Results Summary

**⚠️ CRITICAL UPDATE:** Comprehensive testing with 100 sentences reveals **catastrophic performance degradation** at scale:

| Metric | 15 Sentences | 100 Sentences | Change |
|--------|--------------|---------------|---------|
| **Manual Labels Accuracy** | 100% | **49%** | **-51%** 🔴 |
| **K-Means Accuracy** | 86.67% | **40%** | **-46.67%** 🔴 |
| **Category A Accuracy** | - | **100%** | ✅ Perfect |
| **Category B Accuracy** | - | **17.5%** | 🔴 Catastrophic |
| **Category C Accuracy** | - | **10%** | 🔴 Catastrophic |

**Key Finding:** The system achieves **100% accuracy for Hope/Aspiration (Category A)** but **catastrophically fails for Conflict/Violence (17.5%) and Science/Technology (10%)**. 93% of all predictions are Category A, regardless of actual content.

**Conclusion:** While promising on small test sets, this approach **fails at scale** due to:
1. **Severe Category A bias** - 93/100 predictions are A
2. **TF-IDF semantic limitations** - Cannot distinguish hope from violence
3. **Insufficient training data** - 9-11 samples per category too small
4. **K-Means clustering failure** - Unstable, semantically meaningless clusters

**Status:** ⚠️ **NOT PRODUCTION READY** - Requires 10x more training data and semantic embeddings (SBERT/transformers)

### 📈 Comparative Analysis: Small vs Large Scale Testing

The most striking discovery of this project is the **dramatic performance collapse** when scaling from a small to larger test set. This comparison reveals critical insights about ML system validation:

**15 Sentences vs 100 Sentences:**

The initial 15-sentence test created a **false sense of success** with 100% accuracy for manual labels and 86.67% for K-Means. However, expanding to 100 sentences exposed fundamental flaws that were hidden by the small sample size. The manual label accuracy plummeted by 51 percentage points to just 49%, while K-Means fell to 40%—barely better than random guessing (33%).

**What went wrong?** The 15-sentence test set happened to contain vocabulary that aligned well with the training data, masking three critical problems: (1) the classifier's severe bias toward Category A, which only became apparent when testing more diverse sentences; (2) TF-IDF's inability to distinguish semantic meaning, as violence and hope sentences share space exploration vocabulary; and (3) insufficient training data, with only 9-11 samples per category unable to cover the vocabulary diversity needed for 100 test sentences.

**The category-level breakdown is particularly revealing:** While Category A (Hope/Aspiration) maintained perfect 100% accuracy across both test sizes, Categories B (Conflict/Violence) and C (Science/Technology) experienced catastrophic failures at scale. Category B accuracy dropped from presumably high levels to just 17.5%, with 82.5% of violence sentences being misclassified as aspirational. Category C fared even worse at 10% accuracy, with 90% of technical sentences incorrectly labeled as Category A. This pattern—93% of all predictions being Category A regardless of content—indicates the classifier learned to predict the majority class rather than distinguishing semantic differences.

**K-Means clustering proved fundamentally unstable.** The cluster structure completely reversed between test runs: at 15 sentences, cluster α dominated with 83% of samples, but at 100 sentences, cluster β absorbed 77%. This instability stems from vectorizing training and test data together, meaning test set size affects cluster formation—a critical flaw for any production system. The alignment accuracy remained consistently poor (43-47%) across both scales, confirming that K-Means cannot discover meaningful semantic categories in TF-IDF space.

**The lesson:** Small test sets are dangerously misleading. The 15-sentence results suggested a working system ready for deployment, but 100 sentences revealed it was fundamentally broken. This demonstrates why rigorous scale testing, cross-validation, and per-category metrics are essential before deploying ML systems. Perfect accuracy on a small test set means nothing if the system collapses when faced with real-world data diversity.

### 🔬 Cross-Domain Validation: War and Peace Comparison

**NEW: Does the source book affect performance?** To answer this, we tested identical experiments on "War and Peace" by Leo Tolstoy (historical military fiction) vs "The Sparrow" (science fiction):

| Dataset | 15 Sent (Manual) | 100 Sent (Manual) | Cat A | Cat B | Cat C | Cluster Stable? |
|---------|------------------|-------------------|-------|-------|-------|-----------------|
| **The Sparrow** | 100% | 49% | 100% | 17.5% | 10% | ❌ No (flip) |
| **War & Peace** | 100% | **63%** ✅ | 85% | **55%** ✅ | **20%** ✅ | ✅ Yes |

**KEY FINDING: War and Peace outperforms by +14% overall accuracy** with dramatic improvements in violence detection (+37.5%) and technical language (+10%).

**Why War and Peace performs better:**
1. **Vocabulary distinctiveness** - Military terms ("bayonet," "grapeshot") clearly signal violence; court language ("prayed," "dreamed") signals hope
2. **Less cross-category overlap** - Space exploration vocabulary appears across all Sparrow categories; military context separates better
3. **Cluster stability** - War and Peace maintains same cluster structure at both scales; Sparrow experiences catastrophic cluster flip
4. **Explicit vs implicit sentiment** - War violence is explicit; Sparrow violence mixed with space exploration metaphors

**Conclusion:** **Domain selection is a critical ML design decision**, not an afterthought. TF-IDF performs better on historical/military domains with distinct vocabulary than science fiction with shared technical terms. Before deploying any TF-IDF classifier, run cross-domain validation—if performance varies >15% (as seen here), upgrade to semantic embeddings.

### 💡 Why This Matters

This project provides **critical lessons for ML practitioners**:

1. **Small test sets are misleadingly optimistic** - 100% accuracy on 15 sentences collapsed to 49% on 100 sentences
2. **TF-IDF fails for semantic classification** - Word overlap ≠ semantic similarity (violence and hope sentences share vocabulary)
3. **Training data quantity matters more than algorithm** - 30 samples insufficient for 3-class k-NN
4. **Category bias detection is essential** - System predicts Category A 93% of time regardless of content
5. **K-Means clustering unsuitable for supervised learning** - Cluster structure unstable across test sizes
6. **Educational value in failure** - Documenting what doesn't work is as valuable as showing what does

### 📊 Analysis Reports

- **[docs/full_analysis_100_sentences.md](docs/full_analysis_100_sentences.md)** - Comprehensive 100-sentence deep dive with root cause analysis
- **[docs/results_analysis.md](docs/results_analysis.md)** - Original 15-sentence analysis report (updated with comparisons)
- **[docs/execution_log_100.txt](docs/execution_log_100.txt)** - Full console output from 100-sentence run

## Author
KobyLev

## Features
- **Modular Architecture**: Clean separation of concerns across 8 specialized modules
- **TF-IDF Vectorization**: Efficient text feature extraction with L2 normalization
- **K-Means Clustering**: Unsupervised pattern discovery with 3 clusters (α, β, γ)
- **k-NN Classification**: Supervised learning with configurable k-value (default: k=5)
- **Comparative Analysis**: Side-by-side evaluation of unsupervised vs supervised approaches
- **Rich Visualizations**:
  - 2D PCA projections of clusters and manual labels
  - Cluster distribution and imbalance analysis
  - Confusion matrices for prediction evaluation
  - Detailed k-NN neighbor analysis
- **Token Counting**: Approximate Claude API usage and cost estimation
- **Comprehensive Reporting**: Detailed accuracy metrics, alignment analysis, and conclusions

## Project Structure
```
L16/
├── main.py                   # Main pipeline orchestrator
├── data.py                   # Dataset definitions (training & test sentences)
├── vectorization.py          # TF-IDF vectorization and normalization
├── clustering.py             # K-Means clustering operations
├── classification.py         # k-NN classification functions
├── analysis.py               # Accuracy metrics and result analysis
├── visualization.py          # Plotting and graph generation
├── utils.py                  # Configuration, token counting, utilities
├── requirements.txt          # Python dependencies
├── .env                      # API keys (create from .env.example)
├── .gitignore                # Git ignore rules
└── docs/                     # Documentation & outputs folder
    ├── full_analysis_100_sentences.md  # 📊 Comprehensive 100-sentence analysis
    ├── results_analysis.md             # Original 15-sentence results (updated)
    ├── execution_log_100.txt           # Console output from 100-sentence run
    ├── planning.md                     # Project planning
    ├── prd.md                          # Product Requirements Document v2.0
    ├── prompt_llm.md                   # LLM interaction logs
    ├── sentiment_analysis_results.png  # 6-panel visualization dashboard
    └── detailed_clustering_analysis.png # 3-panel clustering deep dive
```

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies:
  - scikit-learn >= 1.3.0
  - numpy >= 1.24.0
  - matplotlib >= 3.7.0
  - tiktoken >= 0.5.0
  - python-dotenv >= 1.0.0

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Create a `.env` file for API configuration:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Running the Pipeline
Execute the main script to run the complete analysis:
```bash
python main.py
```

When prompted, enter the number of test sentences to use (default: 100, max: 120).

### Pipeline Steps
The pipeline executes 9 steps automatically:

1. **Data Setup** - Load training (30 sentences) and test datasets
2. **Vectorization** - Convert text to TF-IDF vectors with L2 normalization
3. **K-Means Clustering** - Perform unsupervised clustering (K=3)
4. **k-NN Classification** - Train classifiers on both cluster and manual labels
5. **Results Summary** - Display predictions and comparison tables
6. **Final Analysis** - Evaluate which approach performs better
7. **Token Usage Report** - Display token counts and estimated API costs
8. **Visualization** - Generate main analysis graphs
9. **Detailed Analysis** - Generate clustering deep-dive visualizations

### Output Files
The pipeline generates two visualization files:
- `sentiment_analysis_results.png` - 6-panel overview with clusters, distributions, and accuracy
- `detailed_clustering_analysis.png` - 3-panel deep-dive into K-Means performance

## Dataset
**Training Set**: 30 manually labeled sentences across three categories:
- **Category A**: Hope/Aspiration (spiritual journey, discovery, dreams)
- **Category B**: Conflict/Violence (attacks, battles, suffering)
- **Category C**: Science/Technology (engineering, systems, technical details)

**Test Set**: 120 sentences (40 per category) available for evaluation

All sentences are inspired by themes from Mary Doria Russell's science fiction novel "The Sparrow".

## Methodology

### Vectorization
- TF-IDF (Term Frequency-Inverse Document Frequency) with 100 features
- English stop words removed
- L2 normalization for consistent scaling

### K-Means Clustering
- 3 clusters (matching the 3 manual categories)
- Random state = 42 for reproducibility
- Cluster-to-category alignment analysis

### k-NN Classification
- k=5 neighbors (majority voting)
- Two approaches compared:
  1. Trained on K-Means cluster labels (unsupervised)
  2. Trained on manual category labels (supervised)

## Results Interpretation

### What the Visualizations Show

**Main Results Dashboard** (`sentiment_analysis_results.png`):
- **Top Row**: Side-by-side comparison of K-Means clusters vs manual labels in 2D PCA space
- **Middle Row**: Distribution charts revealing cluster imbalance
- **Bottom Row**: Accuracy comparison and confusion matrix

**Detailed Clustering Analysis** (`detailed_clustering_analysis.png`):
- **Left Panel**: K-Means cluster centers and sample distribution showing 83% concentration in one cluster
- **Middle Panel**: Imbalance ratio warning (12.5:1)
- **Right Panel**: k-NN neighbor analysis showing how predictions are made

### Understanding the Metrics

The pipeline evaluates:
- **Alignment Accuracy**: How well K-Means clusters match manual categories (46.67% - poor alignment)
- **Classification Accuracy**: Performance on test set for both approaches (Manual: 100%, K-Means: 86.67%)
- **Cluster Imbalance**: Distribution of samples across clusters (25-2-3 split indicates severe imbalance)
- **Neighbor Analysis**: Visualization of k-NN decision boundaries and voting patterns

### Deep Dive Analysis
For comprehensive insights including:
- Token usage and cost analysis
- Neighbor voting breakdowns
- Why K-Means failed for this task
- Production recommendations

**📖 Read the full report: [docs/results_analysis.md](docs/results_analysis.md)**

## Documentation

All project documentation is located in the [`docs/`](docs/) folder:

### Analysis Reports
- **[full_analysis_100_sentences.md](docs/full_analysis_100_sentences.md)** - 📊 **16-section comprehensive analysis** of 100-sentence test results including:
  - Root cause analysis of catastrophic failures
  - Category-level accuracy breakdowns (100%, 17.5%, 10%)
  - Cluster instability investigation
  - Statistical significance testing
  - 12 actionable recommendations for production readiness
  - Detailed prediction tables with error patterns

- **[results_analysis.md](docs/results_analysis.md)** - Original 15-sentence analysis (updated with 100-sentence comparisons)

- **[execution_log_100.txt](docs/execution_log_100.txt)** - Complete console output from 100-sentence pipeline execution

#### War and Peace Dataset (Cross-Domain Validation)
- **[docs/cross_book_comparative_analysis.md](docs/cross_book_comparative_analysis.md)** - 📚 **Complete cross-domain study** comparing The Sparrow vs War and Peace
- **[docs/second_execution_log_15.txt](docs/second_execution_log_15.txt)** - War and Peace 15-sentence run (100% accuracy both approaches)
- **[docs/second_execution_log_100.txt](docs/second_execution_log_100.txt)** - War and Peace 100-sentence run (63% manual, 47% K-Means)
- **Quick reference:** See Section 17 in full_analysis_100_sentences.md for integrated comparison

### Project Documentation
- **[planning.md](docs/planning.md)** - Project planning and implementation roadmap
- **[prd.md](docs/prd.md)** - Product Requirements Document (v2.0 - comprehensive with all requirements and outcomes)
- **[prompt_llm.md](docs/prompt_llm.md)** - LLM prompts and AI interaction logs

### Visualizations
- **[sentiment_analysis_results.png](docs/sentiment_analysis_results.png)** - 6-panel dashboard (clusters, distributions, accuracy, confusion matrix)
- **[detailed_clustering_analysis.png](docs/detailed_clustering_analysis.png)** - 3-panel deep dive (PCA, imbalance, k-NN neighbors)

## License
MIT
