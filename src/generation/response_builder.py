from PIL import Image
import io
import os
import base64

class ResponseBuilder:
    def __init__(self, llm_interface, data_dir):
        self.llm_interface = llm_interface
        self.data_dir = data_dir
    
    def build_response(self, query, retrieved_items, include_images=True):
        """Build a comprehensive response based on retrieved items."""
        # Prepare context from retrieved items
        context_parts = []
        image_data = []
        
        for item in retrieved_items:
            # Add text content
            if 'content' in item and item['content']:
                source_info = f"[Source: {item['document_id']}, Page {item['page_num']}]"
                context_parts.append(f"{item['content']} {source_info}")
            
            # Collect image info if needed
            if include_images and item.get('chunk_type') == 'image':
                image_info = {
                    'id': item['chunk_id'],
                    'document_id': item['document_id'],
                    'page_num': item['page_num'],
                    'extracted_text': item.get('extracted_text', '')
                }
                image_data.append(image_info)
        
        # Create the context string for the LLM
        context = "\n\n".join(context_parts)
        
        # Prepare the prompt for the LLM
        system_message = """
        You are an aerospace expert assistant. Answer the user's query based on the provided context.
        If the context doesn't contain enough information to answer, say so clearly.
        If relevant images were found, refer to them in your response.
        Provide specific and accurate information, citing sources where appropriate.
        """
        
        prompt = f"""
        User Query: {query}
        
        Context Information:
        {context}
        
        {"Images were also found relevant to this query." if image_data else "No relevant images were found."}
        
        Please provide a comprehensive and accurate response to the query.
        """
        
        # Generate the text response
        text_response = self.llm_interface.generate_response(prompt, system_message)
        
        # Create the final response object
        response = {
            'text_response': text_response,
            'sources': [{'document_id': item['document_id'], 'page_num': item['page_num']} 
                       for item in retrieved_items],
            'images': image_data if include_images else []
        }
        
        return response