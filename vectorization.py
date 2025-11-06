"""
Vectorization Module - Text Processing and Feature Extraction
Handles TF-IDF vectorization and normalization of text data
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def vectorize_and_normalize(training_sentences, test_sentences, max_features=100):
    """
    Vectorize training and test sentences using TF-IDF and normalize.

    Args:
        training_sentences: List of training sentence strings
        test_sentences: List of test sentence strings
        max_features: Maximum number of TF-IDF features (default: 100)

    Returns:
        tuple: (train_vectors, test_vectors, vectorizer)
            - train_vectors: Normalized TF-IDF vectors for training data
            - test_vectors: Normalized TF-IDF vectors for test data
            - vectorizer: Fitted TfidfVectorizer object
    """
    # Combine all sentences for consistent vectorization
    all_sentences = training_sentences + test_sentences

    # Use TfidfVectorizer for efficient vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    all_vectors = vectorizer.fit_transform(all_sentences).toarray()

    # L2 Normalization
    all_vectors_normalized = normalize(all_vectors, norm="l2")

    # Split back into training and test sets
    train_vectors = all_vectors_normalized[:len(training_sentences)]
    test_vectors = all_vectors_normalized[len(training_sentences):]

    return train_vectors, test_vectors, vectorizer


def get_vector_info(train_vectors, test_vectors):
    """
    Get information about vectorized data.

    Args:
        train_vectors: Training vectors array
        test_vectors: Test vectors array

    Returns:
        dict: Dictionary containing vector information
    """
    return {
        'dimensions': train_vectors.shape[1],
        'train_shape': train_vectors.shape,
        'test_shape': test_vectors.shape,
        'normalization': 'L2'
    }
