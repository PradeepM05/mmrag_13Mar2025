from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_document(self, document_data, chunk_size=1000, chunk_overlap=200):
        """Process a document and generate embeddings for text chunks."""
        results = []
        
        for page in document_data['pages']:
            # Process main text content
            text = page['text']
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                embedding = self.model.encode(chunk)
                results.append({
                    'document_id': document_data['metadata']['filename'],
                    'page_num': page['page_num'],
                    'chunk_id': f"page_{page['page_num']}_chunk_{i+1}",
                    'chunk_type': 'text',
                    'content': chunk,
                    'embedding': embedding.tolist()
                })
            
            # Process text from tables
            for i, table in enumerate(page.get('tables', [])):
                if 'text' in table and table['text'].strip():
                    embedding = self.model.encode(table['text'])
                    results.append({
                        'document_id': document_data['metadata']['filename'],
                        'page_num': page['page_num'],
                        'chunk_id': f"page_{page['page_num']}_table_{i+1}",
                        'chunk_type': 'table',
                        'content': table['text'],
                        'embedding': embedding.tolist()
                    })
            
            # Process text from formulas
            for i, formula in enumerate(page.get('formulas', [])):
                if 'text' in formula and formula['text'].strip():
                    embedding = self.model.encode(formula['text'])
                    results.append({
                        'document_id': document_data['metadata']['filename'],
                        'page_num': page['page_num'],
                        'chunk_id': f"page_{page['page_num']}_formula_{i+1}",
                        'chunk_type': 'formula',
                        'content': formula['text'],
                        'embedding': embedding.tolist()
                    })
            
            # Process OCR text from images
            for i, image in enumerate(page.get('images', [])):
                if 'extracted_text' in image and image['extracted_text'].strip():
                    embedding = self.model.encode(image['extracted_text'])
                    results.append({
                        'document_id': document_data['metadata']['filename'],
                        'page_num': page['page_num'],
                        'chunk_id': f"page_{page['page_num']}_img_{i+1}_text",
                        'chunk_type': 'image_text',
                        'content': image['extracted_text'],
                        'embedding': embedding.tolist()
                    })
        
        return results
    
    def _chunk_text(self, text, chunk_size, chunk_overlap):
        """Split text into chunks with overlap."""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            start += (chunk_size - chunk_overlap)
        
        return chunks
