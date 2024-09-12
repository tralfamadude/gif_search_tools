import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def similar_vectors(vectors, threshold=0.999):
    """
    Check if all vectors in a list are similar based on cosine similarity.
    
    Parameters:
    vectors (list of numpy arrays): List of vectors to check.
    threshold (float): Cosine similarity threshold above which vectors are considered similar.
    
    Returns:
    bool: True if all vectors are similar, False otherwise.
    """
    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(vectors)
    
    # Check if all off-diagonal elements in the similarity matrix are above the threshold
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] < threshold:
                return False
    return True

# Example usage:
vectors1 = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])]
vectors2 = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([-1.0, -1.0])]

print(similar_vectors(vectors1))  # Should return True
print(similar_vectors(vectors2))  # Should return False
