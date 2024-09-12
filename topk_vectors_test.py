import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

def select_representative_vectors(embeddings, k, neighborhood_threshold=0.1):
    """
    Selects up to k representative vectors from a set of PyTorch embeddings using k-means clustering.
    Ensures that selected vectors are at least a neighborhood_threshold cosine distance apart.

    Args:
        embeddings (List[torch.Tensor]): A list of PyTorch tensors representing the embeddings.
        k (int): Number of clusters/vectors to pick.
        neighborhood_threshold (float): The minimum cosine distance between returned vectors.

    Returns:
        List[torch.Tensor]: A list of selected representative vectors.
    """
    if not embeddings or k <= 0:
        return []

    # Convert list of embeddings (PyTorch tensors) to numpy array for clustering
    embeddings_np = torch.stack(embeddings).numpy()

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=min(k, len(embeddings)), random_state=42)
    kmeans.fit(embeddings_np)

    # Get cluster centroids
    centroids = kmeans.cluster_centers_

    # Convert centroids back to PyTorch tensors
    centroids_tensors = [torch.tensor(c) for c in centroids]

    # Filter out vectors that are within the neighborhood threshold of each other
    selected_vectors = []
    for centroid in centroids_tensors:
        if all(cosine(centroid.numpy(), selected.numpy()) > neighborhood_threshold for selected in selected_vectors):
            selected_vectors.append(centroid)

    return selected_vectors

# Example usage
if __name__ == "__main__":
    # Sample set of 10 random embeddings of dimension 5
    #torch.manual_seed(42)
    embeddings = [torch.rand(500) for _ in range(60)]
    selected_vectors = select_representative_vectors(embeddings, k=3, neighborhood_threshold=0.025)
    print(f"{len(selected_vectors)} Selected Representative Vectors")
    #for vec in selected_vectors:
    #    print(vec)

