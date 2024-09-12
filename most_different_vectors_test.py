import sys
import os
import json
import torch
import base64
from PIL import Image
from io import BytesIO
import open_clip
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from scipy.spatial.distance import cosine
from collections import defaultdict
from codetiming import Timer
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings


def most_different_vectors(vectors, top_n=3, threshold=0.999):
    """
    Identify up to `top_n` most different vectors based on cosine similarity.
    
    Parameters:
    vectors (list of numpy arrays): List of vectors to check.
    top_n (int): Maximum number of most different vectors to return.
    threshold (float): Cosine similarity threshold to consider vectors as too similar.
    
    Returns:
    list of numpy arrays: List of up to `top_n` most different vectors.
    """
    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(vectors)
    
    # Check if all vectors are similar based on the threshold
    n = len(vectors)
    all_similar = True
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] < threshold:
                all_similar = False
                break
        if not all_similar:
            break
    
    # If all vectors are too similar, return just one vector
    if all_similar:
        return [vectors[0]]
    
    # Convert similarity matrix to dissimilarity (1 - similarity)
    dissimilarity_matrix = 1 - similarity_matrix
    
    # Sum the dissimilarity for each vector to rank them
    dissimilarity_sums = np.sum(dissimilarity_matrix, axis=1)
    
    # Sort vectors by their dissimilarity sums (most dissimilar vectors first)
    sorted_indices = np.argsort(dissimilarity_sums)[::-1]
    
    # Initialize result list with the first most different vector
    result = [vectors[sorted_indices[0]]]
    
    # Now, iteratively add vectors that are dissimilar from those already selected
    for idx in sorted_indices[1:]:
        if len(result) >= top_n:
            break
        
        # Check if the new vector is sufficiently different from all selected vectors
        is_different = True
        for selected_idx in [sorted_indices[0] for sorted_idx in result]:
            # Use precomputed similarity matrix instead of calling cosine_similarity again
            if similarity_matrix[idx, selected_idx] >= threshold:
                is_different = False
                break
        
        if is_different:
            result.append(vectors[idx])
    
    return result


def varname(var):
    for name, value in globals().items():
        if value is var:
            return name
    for name, value in locals().items():
        if value is var:
            return name
    return "UNKOWN"


def most_different_helper(v):
    print(f"vectors_1cluster_identical: {varname(v)} {len(vectors_1cluster_identical(v))}")


vectors_1cluster_identical = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])]
vectors_1cluster_similar = [np.array([1.0, 1.0]), np.array([1.001, 1.001]), np.array([1.0001, 1.0001]), np.array([1.0001, 1.0001])]
vectors_2clusters = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([-1.0, -1.0])]
vectors_2clusters_b = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([-1.0, -1.0]), np.array([-1.01, -1.01])]
vectors_3clusters = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([-1.0, -1.0]), np.array([-1.01, -1.01]) , np.array([-.51, -.51]) ]

most_different_helper(vectors_1cluster_identical)
most_different_helper(vectors_1cluster_similar)
most_different_helper(vectors_2clusters)
most_different_helper(vectors_2clusters_b)
most_different_helper(vectors_3clusters)

