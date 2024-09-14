import numpy as np

def maxmin_diversity(vector_list, k, threshold):
    """
    Selects up to k diverse vectors using the Maxmin algorithm.
    
    Args:
    vector_list (list): List of numpy arrays or lists, each representing a vector
    k (int): Maximum number of vectors to select
    threshold (float): Cosine similarity threshold. If all remaining vectors are within
                       this threshold of similarity to selected vectors, stop selecting.
    
    Returns:
    list: Indices of selected diverse vectors
    """
    n_vectors = len(vector_list)
    
    # Convert list of vectors to 2D numpy array
    vectors = np.array([np.array(v) for v in vector_list])
    
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    
    # Initialize with a random vector
    selected = [np.random.randint(n_vectors)]
    
    while len(selected) < k:
        # Compute cosine similarities between selected and all vectors
        similarities = np.dot(normalized_vectors, normalized_vectors[selected].T)
        
        # Find the maximum similarity for each vector to any selected vector
        max_similarities = np.max(similarities, axis=1)
        
        # Find the vector with the minimum max similarity (i.e., most diverse)
        candidate_idx = np.argmin(max_similarities)
        
        # Check if the candidate is diverse enough
        if 1 - max_similarities[candidate_idx] <= threshold:
            # If not diverse enough, stop selecting
            break
        
        selected.append(candidate_idx)
    
    return selected

# Test cases
if __name__ == "__main__":
    k = 3
    threshold = 0.05
    
    # Helper function to create a slightly perturbed version of a vector
    def perturb(v, scale=0.01):
        v = np.array(v)  # Convert to numpy array if it's a list
        return v + np.random.normal(0, scale, v.size)
    
    # Test case 1: 4 vectors all pointed in almost the same direction
    vectors1 = [[1, 0, 0]] + [perturb([1, 0, 0]) for _ in range(3)]
    
    # Test case 2: 5 vectors clustered in 2 different directions
    vectors2 = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], list(perturb([0, 1, 0]))]
    
    # Test case 3: 10 vectors clustered in 3 different directions
    vectors3 = [[1, 0, 0], [1, 0, 0], [1, 0, 0],
                [0, 1, 0], [0, 1, 0], [0, 1, 0],
                [0, 0, 1], [0, 0, 1], [0, 0, 1],
                list(perturb([0, 0, 1]))]
    
    # Test case 4: 10 vectors clustered in 5 different directions
    vectors4 = [[1, 0, 0], [1, 0, 0],
                [0, 1, 0], [0, 1, 0],
                [0, 0, 1], [0, 0, 1],
                [1, 1, 0], [1, 1, 0],
                [1, 0, 1], [1, 0, 1]]
    
    # Test case 5: 2 vectors pointed in exactly opposite directions
    vectors5 = [[1, 0, 0], [-1, 0, 0]]
    
    test_cases = [vectors1, vectors2, vectors3, vectors4, vectors5]
    
    for i, vectors in enumerate(test_cases, 1):
        result = maxmin_diversity(vectors, k, threshold)
        print(f"\nTest case {i}:")
        print(f"Input vectors:\n{vectors}")
        print(f"Selected diverse vectors (indices): {result}")
        print(f"Number of vectors selected: {len(result)}")
        
        # Print cosine similarities between selected vectors
        selected_vectors = np.array([vectors[i] for i in result])
        if len(selected_vectors) > 1:
            norms = np.linalg.norm(selected_vectors, axis=1, keepdims=True)
            normalized_selected = selected_vectors / norms
            similarities = np.dot(normalized_selected, normalized_selected.T)
            np.fill_diagonal(similarities, 0)  # Zero out self-similarities
            print("Cosine similarities between selected vectors:")
            print(similarities)
        else:
            print("Only one vector selected, no similarities to compute.")

