# Product Requirements Document

## Project Title
Sentiment Analysis with Clustering and Classification

## Author
KobyLev

## Objective
To develop a sentiment analysis system that can categorize sentences into three groups using both unsupervised (K-means) and supervised (KNN) machine learning algorithms.

## Scope
- Generate a dataset of sentences, categorized into three groups (e.g., positive, negative, neutral).
- Implement text vectorization using tokenization and Word2Vec.
- Apply K-means clustering to identify natural groupings in the data.
- Apply KNN classification to predict the category of new sentences.
- Analyze and compare the results of both algorithms.

## Functional Requirements
1. Sentence generation and categorization.
2. Text vectorization.
3. K-means clustering.
4. KNN classification.
5. Results analysis and reporting.

## Non-Functional Requirements
- Code must be well-documented.
- Results must be reproducible.
- The system should be modular and easy to extend.

## Deliverables
- Source code.
- Analysis reports (planning.md, prd.md, readme.md).
- Example dataset.

## Timeline
- Week 1: Dataset preparation and vectorization.
- Week 2: Clustering and classification implementation.
- Week 3: Results analysis and documentation.

## Assumptions
- The dataset will be balanced (equal number of sentences per category).
- The vectorization will be normalized.
- The algorithms will be evaluated using standard metrics.

## Risks
- Insufficient data may lead to poor clustering results.
- The choice of vectorization method may affect performance.
