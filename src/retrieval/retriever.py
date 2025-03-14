import numpy as np

class MultimodalRetriever:
    def __init__(self, vector_store, text_embedder, image_embedder=None):
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
    
    def retrieve(self, query, top_k=5, mode="text", image_bytes=None):
        """
        Retrieve relevant content based on query.
        
        Args:
            query: Text query or image query path
            top_k: Number of results to return
            mode: 'text', 'image', or 'hybrid'
            image_bytes: Raw image bytes if mode is 'image' or 'hybrid'
        """
        results = []
        
        if mode == "text" or mode == "hybrid":
            # Generate text query embedding
            query_embedding = self.text_embedder.generate_embeddings([query])[0]
            
            # Search vector store
            text_results = self.vector_store.search_text(query_embedding, top_k=top_k)
            results.extend(text_results)
        
        if (mode == "image" or mode == "hybrid") and self.image_embedder and image_bytes:
            # Generate image query embedding
            query_embedding = self.image_embedder.generate_embedding(image_bytes)
            
            if query_embedding:
                # Search vector store for similar images
                image_results = self.vector_store.search_images(query_embedding, top_k=top_k)
                results.extend(image_results)
        
        # Sort by similarity and take top_k
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return results[:top_k]
