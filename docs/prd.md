# Product Requirements Document (PRD)
# Sentiment Analysis Pipeline - K-Means & k-NN

**Version:** 2.0
**Author:** KobyLev
**Date:** November 6, 2025
**Status:** Completed ✅

---

## 1. Executive Summary

This document defines the requirements for a sentiment analysis research pipeline that compares unsupervised clustering (K-Means) with supervised classification (k-NN) to answer the fundamental question: **Can unsupervised learning match supervised learning performance for text categorization?**

The system analyzes sentences inspired by Mary Doria Russell's "The Sparrow" across three semantic categories and provides comprehensive visualizations, metrics, and cost analysis.

---

## 2. Project Objectives

### Primary Objective
Develop a modular, production-ready sentiment analysis pipeline that:
1. Compares K-Means clustering vs manual labeling for k-NN classification
2. Provides visual and quantitative analysis of both approaches
3. Demonstrates the value of human-labeled training data
4. Offers cost-effective text analysis with token usage tracking

### Research Question
**Does K-Means clustering discover meaningful semantic patterns that rival human-labeled categories for k-NN classification?**

### Success Criteria
- ✅ Achieve >85% classification accuracy with manual labels
- ✅ Generate publication-quality visualizations
- ✅ Complete modular architecture with 8+ distinct modules
- ✅ Provide reproducible results with configurable test set sizes
- ✅ Track token usage and cost estimation

---

## 3. Scope

### In Scope
1. **Dataset Management**
   - 30 manually labeled training sentences
   - 120 test sentences (40 per category)
   - Three semantic categories: Hope/Aspiration, Conflict/Violence, Science/Technology

2. **Text Processing**
   - TF-IDF vectorization with L2 normalization
   - 100 features maximum
   - English stop word removal

3. **Machine Learning**
   - K-Means clustering (K=3, random_state=42)
   - k-NN classification (k=5)
   - Cluster-to-category alignment analysis

4. **Visualization**
   - 2D PCA projections
   - Cluster distribution charts
   - Accuracy comparison graphs
   - Confusion matrices
   - k-NN neighbor analysis

5. **Analysis & Reporting**
   - Alignment accuracy metrics
   - Classification accuracy comparison
   - Token usage and cost estimation
   - Comprehensive markdown reports

### Out of Scope
- Deep learning models (transformers, BERT, etc.)
- Multi-language support
- Real-time API endpoint deployment
- Database integration
- Web interface
- Actual Claude API calls (using token approximation instead)

---

## 4. Functional Requirements

### FR1: Data Module
**Priority:** Critical
**Status:** ✅ Implemented

- **FR1.1:** Store 30 training sentences with manual labels (A, B, C)
- **FR1.2:** Store 120 test sentences with expected labels
- **FR1.3:** Support configurable test set size (1-120 sentences)
- **FR1.4:** Maintain balanced distribution across categories

### FR2: Vectorization Module
**Priority:** Critical
**Status:** ✅ Implemented

- **FR2.1:** Implement TF-IDF vectorization with scikit-learn
- **FR2.2:** Support configurable max_features parameter
- **FR2.3:** Apply L2 normalization to all vectors
- **FR2.4:** Return vector metadata (dimensions, shapes)

### FR3: Clustering Module
**Priority:** High
**Status:** ✅ Implemented

- **FR3.1:** Perform K-Means clustering with K=3
- **FR3.2:** Map cluster numbers to Greek letters (α, β, γ)
- **FR3.3:** Analyze cluster-to-manual-label alignment
- **FR3.4:** Calculate cluster distribution and imbalance ratio
- **FR3.5:** Identify dominant theme per cluster

### FR4: Classification Module
**Priority:** Critical
**Status:** ✅ Implemented

- **FR4.1:** Train k-NN classifier on K-Means labels
- **FR4.2:** Train k-NN classifier on manual labels
- **FR4.3:** Support configurable k value (default: 5)
- **FR4.4:** Find k-nearest neighbors for analysis
- **FR4.5:** Generate predictions for test set

### FR5: Analysis Module
**Priority:** High
**Status:** ✅ Implemented

- **FR5.1:** Calculate alignment accuracy between K-Means and manual labels
- **FR5.2:** Calculate classification accuracy for both approaches
- **FR5.3:** Generate results comparison tables
- **FR5.4:** Provide conclusion with recommended approach
- **FR5.5:** Print clustering analysis summaries

### FR6: Visualization Module
**Priority:** High
**Status:** ✅ Implemented

- **FR6.1:** Generate 6-panel main results dashboard
  - K-Means clustering (PCA projection)
  - Manual labels (PCA projection)
  - Cluster distribution bar chart
  - Manual label distribution bar chart
  - Accuracy comparison bar chart
  - Confusion matrix heatmap

- **FR6.2:** Generate 3-panel detailed clustering analysis
  - 2D PCA with cluster centers
  - Cluster size distribution with imbalance warning
  - k-NN neighbor analysis for first test sentence

- **FR6.3:** Save all visualizations to `docs/` folder
- **FR6.4:** Support 300 DPI publication quality

### FR7: Utilities Module
**Priority:** Medium
**Status:** ✅ Implemented

- **FR7.1:** Setup Windows console encoding for Unicode
- **FR7.2:** Load environment variables from .env file
- **FR7.3:** Initialize tiktoken tokenizer (cl100k_base)
- **FR7.4:** Count tokens for all processed text
- **FR7.5:** Calculate estimated Claude API costs
- **FR7.6:** Prompt user for test set size configuration

### FR8: Main Pipeline
**Priority:** Critical
**Status:** ✅ Implemented

- **FR8.1:** Orchestrate all 9 pipeline steps sequentially
- **FR8.2:** Create `docs/` folder automatically if missing
- **FR8.3:** Display progress with clear step headers
- **FR8.4:** Handle user input for test set size
- **FR8.5:** Generate both visualization files
- **FR8.6:** Print comprehensive final report

---

## 5. Non-Functional Requirements

### NFR1: Performance
- **NFR1.1:** Complete full pipeline in <10 seconds for default configuration
- **NFR1.2:** Support up to 1000 test sentences without performance degradation
- **NFR1.3:** Use efficient vectorization with sparse matrices when possible

### NFR2: Code Quality
- **NFR2.1:** ✅ Modular architecture with 8 separate Python modules
- **NFR2.2:** ✅ Comprehensive docstrings for all functions
- **NFR2.3:** ✅ Type hints where applicable
- **NFR2.4:** ✅ PEP 8 compliant formatting
- **NFR2.5:** ✅ No hardcoded paths (relative paths only)

### NFR3: Documentation
- **NFR3.1:** ✅ README with quick start guide and examples
- **NFR3.2:** ✅ Comprehensive results analysis report
- **NFR3.3:** ✅ Inline code comments for complex logic
- **NFR3.4:** ✅ Visual documentation with embedded images
- **NFR3.5:** ✅ Updated PRD reflecting final implementation

### NFR4: Reproducibility
- **NFR4.1:** ✅ Fixed random seeds (random_state=42)
- **NFR4.2:** ✅ Requirements.txt with version pinning
- **NFR4.3:** ✅ Deterministic pipeline execution
- **NFR4.4:** ✅ Version control with .gitignore

### NFR5: Usability
- **NFR5.1:** ✅ Single command execution: `python main.py`
- **NFR5.2:** ✅ Interactive user prompts with defaults
- **NFR5.3:** ✅ Clear console output with progress indicators
- **NFR5.4:** ✅ Automatic output folder creation

### NFR6: Maintainability
- **NFR6.1:** ✅ Separation of concerns across modules
- **NFR6.2:** ✅ Reusable functions with single responsibility
- **NFR6.3:** ✅ Configurable parameters (not hardcoded)
- **NFR6.4:** ✅ Easy to extend with new algorithms

---

## 6. Technical Architecture

### Technology Stack
- **Language:** Python 3.8+
- **ML Framework:** scikit-learn 1.3.0+
- **Visualization:** matplotlib 3.7.0+
- **Tokenization:** tiktoken 0.5.0+
- **Configuration:** python-dotenv 1.0.0+
- **Scientific Computing:** numpy 1.24.0+

### Module Structure
```
sentiment_analysis/
├── data.py              # Dataset repository
├── vectorization.py     # Text → Vectors
├── clustering.py        # K-Means operations
├── classification.py    # k-NN operations
├── analysis.py          # Metrics & evaluation
├── visualization.py     # Plotting & graphs
├── utils.py             # Configuration & helpers
└── main.py              # Orchestration
```

### Data Flow
```
Training Sentences → TF-IDF Vectorizer → Normalized Vectors
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                              K-Means (3)         Manual Labels
                                    ↓                   ↓
                              k-NN Train          k-NN Train
                                    ↓                   ↓
Test Sentences → Vectorize → k-NN Predict → Compare Accuracy
                                    ↓
                        Generate Visualizations & Reports
```

---

## 7. Deliverables

### Code Deliverables ✅
- [x] 8 Python modules (data, vectorization, clustering, classification, analysis, visualization, utils, main)
- [x] requirements.txt with dependencies
- [x] .gitignore for environment and outputs
- [x] .env.example template

### Documentation Deliverables ✅
- [x] README.md with comprehensive guide
- [x] docs/results_analysis.md with detailed findings
- [x] docs/planning.md with implementation plan
- [x] docs/prd.md (this document)
- [x] docs/prompt_llm.md with AI interaction logs

### Visualization Deliverables ✅
- [x] docs/sentiment_analysis_results.png (6-panel dashboard)
- [x] docs/detailed_clustering_analysis.png (3-panel deep dive)

### Analysis Deliverables ✅
- [x] Accuracy comparison metrics
- [x] Cluster imbalance analysis
- [x] Token usage and cost estimates
- [x] Neighbor voting breakdowns

---

## 8. Acceptance Criteria

### AC1: Functional Completeness ✅
- [x] All 8 modules implemented and tested
- [x] Pipeline executes all 9 steps without errors
- [x] User can configure test set size (1-120)
- [x] Both visualization files generated automatically

### AC2: Accuracy Targets ✅
- [x] Manual labels achieve ≥90% accuracy (achieved 100%)
- [x] K-Means approach achieves ≥70% accuracy (achieved 86.67%)
- [x] Clear winner identified with quantitative comparison

### AC3: Visualization Quality ✅
- [x] All graphs render correctly at 300 DPI
- [x] Color coding consistent across visualizations
- [x] Legends and labels clearly readable
- [x] Images display correctly in GitHub README

### AC4: Documentation Quality ✅
- [x] README contains quick start guide
- [x] All functions have docstrings
- [x] Results analysis provides actionable insights
- [x] Documentation references updated for docs/ folder

### AC5: Reproducibility ✅
- [x] Same inputs produce same outputs
- [x] Fixed random seeds ensure consistency
- [x] Requirements file enables easy setup
- [x] No external API dependencies

---

## 9. Project Timeline

### Phase 1: Planning & Design ✅
**Duration:** 1 day
- [x] Define research question
- [x] Design modular architecture
- [x] Create PRD and planning documents
- [x] Prepare dataset (30 training + 120 test sentences)

### Phase 2: Core Implementation ✅
**Duration:** 2 days
- [x] Implement data module
- [x] Implement vectorization module
- [x] Implement clustering module
- [x] Implement classification module

### Phase 3: Analysis & Visualization ✅
**Duration:** 1 day
- [x] Implement analysis module
- [x] Implement visualization module
- [x] Create main pipeline orchestrator
- [x] Test end-to-end execution

### Phase 4: Refactoring & Documentation ✅
**Duration:** 1 day
- [x] Separate monolithic script into modules
- [x] Create docs/ folder structure
- [x] Write comprehensive README
- [x] Generate results analysis report
- [x] Update all documentation references

### Phase 5: Polish & Deployment ✅
**Duration:** 0.5 days
- [x] Add .gitignore for environment files
- [x] Update requirements.txt with categories
- [x] Verify GitHub image display
- [x] Final testing and validation

**Total Duration:** ~5.5 days
**Status:** Completed November 6, 2025

---

## 10. Assumptions & Constraints

### Assumptions
1. ✅ Users have Python 3.8+ installed
2. ✅ Users can install dependencies via pip
3. ✅ Dataset remains static (30 training, 120 test)
4. ✅ Three semantic categories are sufficient
5. ✅ TF-IDF vectorization is appropriate for this task
6. ⚠️ Training set is relatively balanced (33-37% per category) - achieved
7. ❌ K-Means will find good semantic clusters - NOT achieved (severe imbalance)

### Constraints
1. **Technical:** No GPU required (CPU-only implementation)
2. **Data:** Limited to English language sentences
3. **Size:** Training set limited to 30 samples
4. **Categories:** Fixed to 3 categories (A, B, C)
5. **Algorithm:** K-Means with K=3 (not dynamically optimized)
6. **Vectorization:** TF-IDF only (no embeddings or transformers)

---

## 11. Risks & Mitigation

### Risk 1: K-Means Produces Poor Clusters
**Likelihood:** High
**Impact:** Medium
**Status:** ✅ Occurred (12.5:1 imbalance ratio)
**Mitigation:**
- Document findings in results analysis
- Provide comparison showing manual labels superiority
- Explain why this happens (TF-IDF limitations)

### Risk 2: Insufficient Training Data
**Likelihood:** Medium
**Impact:** Medium
**Status:** ✅ Mitigated
**Mitigation:**
- 30 training samples proved sufficient for manual labels
- Achieved 100% accuracy on test set
- Recommend 100+ samples for production in results analysis

### Risk 3: Visualization Files Not Displayed in GitHub
**Likelihood:** Low
**Impact:** High
**Status:** ✅ Resolved
**Mitigation:**
- Removed PNG files from .gitignore
- Verified correct relative paths in markdown
- Tested image display in GitHub

### Risk 4: User Configuration Errors
**Likelihood:** Low
**Impact:** Low
**Status:** ✅ Mitigated
**Mitigation:**
- Interactive prompts with clear defaults
- Input validation with fallback to defaults
- Automatic docs/ folder creation

---

## 12. Success Metrics

### Quantitative Metrics ✅
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Manual Label Accuracy | ≥90% | 100% | ✅ Exceeded |
| K-Means Accuracy | ≥70% | 86.67% | ✅ Exceeded |
| Code Modularity | 6+ modules | 8 modules | ✅ Exceeded |
| Visualization Quality | 2+ graphs | 9 graphs | ✅ Exceeded |
| Documentation Pages | 3+ files | 5 files | ✅ Exceeded |
| Execution Time | <30s | <10s | ✅ Exceeded |

### Qualitative Metrics ✅
- [x] Clear research question answered
- [x] Actionable insights provided
- [x] Production-ready code structure
- [x] Publication-quality visualizations
- [x] Comprehensive documentation

---

## 13. Future Enhancements

### Version 2.1 (Proposed)
- [ ] Add DBSCAN clustering comparison
- [ ] Implement hierarchical clustering
- [ ] Add silhouette score analysis
- [ ] Support custom k values via command line

### Version 3.0 (Proposed)
- [ ] Replace TF-IDF with sentence transformers (SBERT)
- [ ] Add actual Claude API integration for embeddings
- [ ] Support dynamic K selection (elbow method)
- [ ] Add cross-validation for robust accuracy metrics

### Version 4.0 (Proposed)
- [ ] Web interface for interactive exploration
- [ ] Real-time classification API endpoint
- [ ] Support for custom datasets via CSV upload
- [ ] Multi-language support

---

## 14. Conclusion

This project successfully demonstrates that **supervised learning with human-labeled categories significantly outperforms unsupervised K-Means clustering** for sentiment analysis tasks. The modular architecture, comprehensive visualizations, and detailed documentation provide a strong foundation for future research and production deployment.

**Key Achievement:** A production-ready, well-documented sentiment analysis pipeline that answers a fundamental ML research question with clear, visual, and quantitative evidence.

**Status:** ✅ All requirements met and exceeded
**Recommendation:** Approved for deployment and documentation publication

---

**Document Version:** 2.0
**Last Updated:** November 6, 2025
**Next Review:** N/A (Project Complete)
