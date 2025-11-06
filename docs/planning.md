# Project Planning

## Author
KobyLev

## Project Overview
This document outlines the planning for the sentiment analysis project, including tasks, milestones, and responsibilities.

## Tasks
1. **Dataset Preparation**
   - Generate sentences for three categories.
   - Ensure balanced dataset.
2. **Text Vectorization**
   - Implement tokenization.
   - Apply Word2Vec for vectorization.
   - Normalize vectors.
3. **Clustering (K-means)**
   - Run K-means on the dataset.
   - Analyze clustering results.
4. **Classification (KNN)**
   - Train KNN on the dataset.
   - Evaluate classification performance.
5. **Results Analysis**
   - Compare clustering and classification results.
   - Document findings.
6. **Documentation**
   - Write detailed reports (readme.md, prd.md, planning.md).

## Milestones
- **Milestone 1:** Dataset preparation and vectorization (Week 1).
- **Milestone 2:** Clustering and classification implementation (Week 2).
- **Milestone 3:** Results analysis and documentation (Week 3).

## Responsibilities
- KobyLev: Project management, coding, analysis, documentation.

## Timeline
- Week 1: Dataset preparation and vectorization.
- Week 2: Clustering and classification implementation.
- Week 3: Results analysis and documentation.

## Risks and Mitigation
- **Risk:** Insufficient data may lead to poor clustering results.
  - **Mitigation:** Ensure balanced dataset and consider data augmentation.
- **Risk:** Choice of vectorization method may affect performance.
  - **Mitigation:** Experiment with different vectorization methods and compare results.

## References
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [gensim documentation](https://radimrehurek.com/gensim/)
