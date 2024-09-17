import unittest
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings

#warnings.filterwarnings("ignore", category=ConvergenceWarning)  # supress warnings if clusters < k for Kmeans

# KMeans method testing here

# The select_representative_vectors function as defined previously
def select_representative_vectors(embeddings, k, neighborhood_threshold=0.1):
    if not embeddings or k <= 0:
        return []

    # Convert embeddings to CPU before converting to numpy
    embeddings_np = torch.stack(embeddings).cpu().numpy()
    
    kmeans = KMeans(n_clusters=min(k, len(embeddings)), random_state=42)
    kmeans.fit(embeddings_np)
    centroids = kmeans.cluster_centers_
    centroids_tensors = [torch.tensor(c) for c in centroids]

    selected_vectors = []
    for centroid in centroids_tensors:
        if all(cosine(centroid.numpy(), selected.numpy()) > neighborhood_threshold for selected in selected_vectors):
            selected_vectors.append(centroid)

    return selected_vectors


class TestSelectRepresentativeVectors(unittest.TestCase):

    def test_empty_embeddings(self):
        embeddings = []
        result = select_representative_vectors(embeddings, k=3)
        self.assertEqual(len(result), 0, "Empty embeddings should return an empty list")

    def test_single_embedding(self):
        embeddings = [torch.tensor([1.0, 0.0, 0.0])]
        result = select_representative_vectors(embeddings, k=3)
        self.assertEqual(len(result), 1, "Single embedding should return a list with one element")
        self.assertTrue(all(result[0].numpy() == embeddings[0].numpy()),
                                  "The single embedding should be returned as-is")

    def test_k_greater_than_embeddings(self):
        embeddings = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])]
        result = select_representative_vectors(embeddings, k=5)
        self.assertEqual(len(result), len(embeddings),
                         "If k is greater than the number of embeddings, all embeddings should be returned")

    def test_identical_embeddings(self):
        embeddings = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])]
        result = select_representative_vectors(embeddings, k=2, neighborhood_threshold=0.1)
        self.assertEqual(len(result), 1,
                         "Identical embeddings should return only one representative vector when threshold is set")

    def test_distinct_embeddings(self):
        embeddings = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0])
        ]
        result = select_representative_vectors(embeddings, k=2, neighborhood_threshold=0.5)
        self.assertEqual(len(result), 2,
                         "The function should return exactly 2 distinct vectors when k=2 and the vectors are far apart")

    def test_threshold_elimination(self):
        embeddings = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.99, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0])
        ]
        result = select_representative_vectors(embeddings, k=3, neighborhood_threshold=0.01)
        self.assertEqual(len(result), 2,
                         "Embeddings within the threshold should be reduced to one representative vector")


if __name__ == '__main__':
    unittest.main()
