from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io

class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def generate_embedding(self, image_bytes):
        """Generate embedding for an image."""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Process image for CLIP
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Convert to list and return
            return image_features.cpu().numpy()[0].tolist()
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return None
    
    def embed_document_images(self, document_data):
        """Process a document and generate embeddings for all images."""
        results = []
        
        for page in document_data['pages']:
            for i, image in enumerate(page.get('images', [])):
                if 'image_bytes' not in image:
                    continue
                
                embedding = self.generate_embedding(image['image_bytes'])
                
                if embedding:
                    results.append({
                        'document_id': document_data['metadata']['filename'],
                        'page_num': page['page_num'],
                        'chunk_id': f"page_{page['page_num']}_img_{i+1}",
                        'chunk_type': 'image',
                        'width': image.get('width'),
                        'height': image.get('height'),
                        'extracted_text': image.get('extracted_text', ''),
                        'embedding': embedding
                    })
        
        return results