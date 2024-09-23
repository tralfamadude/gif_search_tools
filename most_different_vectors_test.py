import numpy as np
import random
import sys

def most_different_vectors(vector_list, k=3, threshold=0.05):
    """
    Selects up to k diverse vectors using the Maxmin algorithm. 
    Complexity: O(kn)
    Does not work on zero vector like [0.0, 0.0]
    
    Args:
    vector_list (list): List of numpy arrays or lists, each representing a vector
    k (int): Maximum number of vectors to select
    threshold (float): Cosine similarity threshold. If all remaining vectors are within
                       this threshold of similarity to selected vectors, stop selecting.
    
    Returns:
    list of numpy arrays: List of up to `k` most different vectors.
    list of indices of `k` most different vectors.
    """
    n_vectors = len(vector_list)
    print(f"DEBUG: input vector_list={vector_list}")
    # Convert list of vectors to 2D numpy array
    vectors = np.array([np.array(v) for v in vector_list])
    
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    print(f"DEBUG: norms={norms}")
    if norms.all() == 0:
        print(f"zero vector found: {vector_list}")
        return [], []
    normalized_vectors = vectors / norms
    print(f"DEBUG: normalized_vectors={normalized_vectors}")
    
    # Initialize with a random vector
    selected = [np.random.randint(n_vectors)]
    print(f"DEBUG: selected={selected}")
    
    while len(selected) < k:
        # Compute cosine similarities between selected and all vectors
        similarities = np.dot(normalized_vectors, normalized_vectors[selected].T)
        print(f"DEBUG: similarities={similarities}")
        
        # Find the maximum similarity for each vector to any selected vector
        max_similarities = np.max(similarities, axis=1)
        print(f"DEBUG: max_similarities={max_similarities}")

        # Mask out already selected indices
        max_similarities[selected] = np.inf
        
        # Find the vector with the minimum max similarity (i.e., most diverse)
        candidate_idx = np.argmin(max_similarities)
        print(f"DEBUG: candidate_idx={candidate_idx}")
        
        # Check if the candidate is diverse enough
        if 1 - max_similarities[candidate_idx] <= threshold:
            # If not diverse enough, stop selecting
            print(f"candidate not diverse enough {max_similarities[candidate_idx]}") # DEBUG
            break
        
        selected.append(candidate_idx)
        
    #print(f"DEBUG: vector_list={vector_list}")
    #print(f"DEBUG: returning={[vector_list[i] for i in selected]}, {selected}")
    return [vector_list[i] for i in selected], selected


def perturb(v, scale, min_value, max_value):
    """Perturbs values in a list within a specified range.

    Args:
      v: A list of floats.
      scale: The maximum perturbation amount.
      min_value: The minimum allowed value.
      max_value: The maximum allowed value.

    Returns:
      A list of perturbed floats.
    """
    perturbed_values = []
    for value in v:
        perturbation = random.uniform(-scale, scale)
        new_value = value + perturbation
        new_value = max(min(new_value, max_value), min_value)
        perturbed_values.append(new_value)
    return perturbed_values


# Test cases
if __name__ == "__main__":
    # first test perturb_values()
    v = [1.0, -1.0, 0.0]
    print(f"Perturb test:  {v}  =>  {perturb(v, 0.02, -1.0, 1.0)}")
    v = [0.0, 0.0, 0.0]
    print(f"Perturb test:  {v}  =>  {perturb(v, 0.02, -1.0, 1.0)}")
    v = [1.0, 1.0, 1.0]
    print(f"Perturb test:  {v}  =>  {perturb(v, 0.02, -1.0, 1.0)}")

    
    k = 3
    threshold = 0.05
    
    
    # Test case 1: 4 vectors all pointed in almost the same direction
    vectors1 = [[1, 0, 0]] + [perturb([1, 0, 0], 0.02, -1.0, 1.0) for _ in range(3)]
    print(f"vectors1={vectors1}")
    
    # Test case 2: 5 vectors clustered in 2 different directions
    vectors2 = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], list(perturb([0, 1, 0], 0.02, -1.0, 1.0))]
    
    # Test case 3: 10 vectors clustered in 3 different directions
    vectors3 = [[1, 0, 0], [1, 0, 0], [1, 0, 0],
                [0, 1, 0], [0, 1, 0], [0, 1, 0],
                [0, 0, 1], [0, 0, 1], [0, 0, 1],
                list(perturb([0, 0, 1], 0.02, -1.0, 1.0))]
    
    # Test case 4: 10 vectors clustered in 5 different directions
    vectors4 = [[1, 0, 0], [1, 0, 0],
                [0, 1, 0], [0, 1, 0],
                [0, 0, 1], [0, 0, 1],
                [1, 1, 0], [1, 1, 0],
                [1, 0, 1], [1, 0, 1]]
    
    # Test case 5: 2 vectors pointed in exactly opposite directions
    vectors5 = [[1, 0, 0], [-1, 0, 0]]

    # Test case 6: more than 3 exactly duplicated vectors
    vectors6 = [[1, -1, 1],[1, -1, 1],[1, -1, 1],[1, -1, 1],[1, -1, 1],[1, -1, 1],[1, -1, 1]]

    # Test case 7: many 0 vectors
    vectors7 = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

    test_cases = [vectors1, vectors2, vectors3, vectors4, vectors5, vectors6, vectors7]
    len_cases = [1, 2, 3, 3, 2, 1, 0]
    
    for i, vectors in enumerate(test_cases, 1):
        result, indices = most_different_vectors(vectors, k, threshold)
        print("\n")
        print(f"\nTest case {i}:")
        print(f"Input vectors:\n{vectors}")
        print(f"Selected diverse vectors (indices): {indices}")
        print(f"Selected diverse result={result}")
        print(f"Number of vectors selected: {len(result)}")
        if len(result)!=len_cases[i-1]:
            print(f"MISMATCH: len={len(result)} vs expected={len_cases[i-1]}")
        
