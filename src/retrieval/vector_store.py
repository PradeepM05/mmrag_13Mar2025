import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVectorStore:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        self.text_index_file = os.path.join(storage_dir, "text_index.json")
        self.image_index_file = os.path.join(storage_dir, "image_index.json")
        self.text_vectors = []
        self.image_vectors = []
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing indices if they exist
        self._load_indices()
    
    def _load_indices(self):
        """Load existing vector indices if available."""
        if os.path.exists(self.text_index_file):
            with open(self.text_index_file, 'r') as f:
                self.text_vectors = json.load(f)
        
        if os.path.exists(self.image_index_file):
            with open(self.image_index_file, 'r') as f:
                self.image_vectors = json.load(f)
    
    def _save_indices(self):
        """Save vector indices to disk."""
        with open(self.text_index_file, 'w') as f:
            json.dump(self.text_vectors, f)
        
        with open(self.image_index_file, 'w') as f:
            json.dump(self.image_vectors, f)
    
    def add_text_vectors(self, vectors):
        """Add text vectors to the index."""
        self.text_vectors.extend(vectors)
        self._save_indices()
    
    def add_image_vectors(self, vectors):
        """Add image vectors to the index."""
        self.image_vectors.extend(vectors)
        self._save_indices()
    
    def search_text(self, query_vector, top_k=5):
        """Search for similar text vectors."""
        if not self.text_vectors:
            return []
        
        # Extract embeddings for similarity calculation
        embeddings = np.array([item['embedding'] for item in self.text_vectors])
        query_vector = np.array(query_vector).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = self.text_vectors[idx].copy()
            result['similarity'] = float(similarities[idx])
            results.append(result)
        
        return results
    
    def search_images(self, query_vector, top_k=5):
        """Search for similar image vectors."""
        if not self.image_vectors:
            return []
        
        # Extract embeddings for similarity calculation
        embeddings = np.array([item['embedding'] for item in self.image_vectors])
        query_vector = np.array(query_vector).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = self.image_vectors[idx].copy()
            result['similarity'] = float(similarities[idx])
            results.append(result)
        
        return results