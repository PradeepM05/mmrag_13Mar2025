import gradio as gr
import os
import tempfile
import json
from PIL import Image
import io
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our components
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.ocr import OCREngine
from src.embedding.text_embedder import TextEmbedder
from src.embedding.image_embedder import ImageEmbedder
from src.retrieval.vector_store import SimpleVectorStore
from src.retrieval.retriever import MultimodalRetriever
from src.generation.llm_interface import LLMInterface
from src.generation.response_builder import ResponseBuilder

class AerospaceRAGApp:
    def __init__(self, data_dir="./data"):
        # Create necessary directories
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.embeddings_dir = os.path.join(data_dir, "embeddings")
        
        for dir_path in [self.data_dir, self.raw_dir, self.processed_dir, self.embeddings_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize components
        print("Initializing components...")
        self.ocr_engine = OCREngine()
        self.pdf_processor = PDFProcessor(ocr_engine=self.ocr_engine)
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.vector_store = SimpleVectorStore(self.embeddings_dir)
        self.retriever = MultimodalRetriever(
            self.vector_store, 
            self.text_embedder, 
            self.image_embedder
        )
        self.llm_interface = LLMInterface()
        self.response_builder = ResponseBuilder(self.llm_interface, self.data_dir)
        
        # Automatically process any PDFs in the raw directory
        self.process_existing_pdfs()
    
    def process_existing_pdfs(self):
        """Automatically process any PDF files in the raw directory that haven't been processed yet."""
        # Get all PDF files in the raw directory
        pdf_files = [f for f in os.listdir(self.raw_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found in the raw directory.")
            return
        
        # Check which files have already been processed
        processed_files = [os.path.splitext(f)[0] for f in os.listdir(self.processed_dir) 
                          if f.endswith('.json')]
        
        # Filter to only unprocessed files
        new_pdf_files = [f for f in pdf_files 
                        if os.path.splitext(f)[0] not in processed_files]
        
        if not new_pdf_files:
            print("All PDF files have already been processed.")
            return
        
        print(f"Found {len(new_pdf_files)} new PDF files in raw directory. Processing...")
        
        # Process each new PDF file
        for pdf_file in new_pdf_files:
            pdf_path = os.path.join(self.raw_dir, pdf_file)
            print(f"Processing {pdf_file}...")
            
            try:
                # Process PDF and extract content
                document_data = self.pdf_processor.process_pdf(pdf_path)
                
                # Save processed data
                processed_file = os.path.join(
                    self.processed_dir, 
                    f"{os.path.splitext(os.path.basename(pdf_file))[0]}.json"
                )
                
                # Remove image bytes before saving (too large)
                save_data = document_data.copy()
                for page in save_data['pages']:
                    for img in page.get('images', []):
                        if 'image_bytes' in img:
                            img['image_bytes'] = '[BINARY DATA REMOVED FOR STORAGE]'
                
                with open(processed_file, "w") as f:
                    json.dump(save_data, f)
                
                # Generate text embeddings
                text_vectors = self.text_embedder.embed_document(document_data)
                self.vector_store.add_text_vectors(text_vectors)
                
                # Generate image embeddings
                image_vectors = self.image_embedder.embed_document_images(document_data)
                self.vector_store.add_image_vectors(image_vectors)
                
                print(f"Successfully processed {pdf_file} with {len(document_data['pages'])} pages")
                print(f"Generated {len(text_vectors)} text vectors and {len(image_vectors)} image vectors")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
    
    def ingest_document(self, file_obj):
        """Process and ingest a document into the RAG system."""
        if file_obj is None:
            return "No file provided."
            
        try:
            # Save uploaded file temporarily
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file_obj.name)
            
            with open(temp_path, "wb") as f:
                f.write(file_obj.read())
            
            # Process PDF and extract content
            document_data = self.pdf_processor.process_pdf(temp_path)
            
            # Save processed data
            processed_file = os.path.join(
                self.processed_dir, 
                f"{os.path.splitext(os.path.basename(file_obj.name))[0]}.json"
            )
            
            # Remove image bytes before saving (too large)
            save_data = document_data.copy()
            for page in save_data['pages']:
                for img in page.get('images', []):
                    if 'image_bytes' in img:
                        img['image_bytes'] = '[BINARY DATA REMOVED FOR STORAGE]'
            
            with open(processed_file, "w") as f:
                json.dump(save_data, f)
            
            # Generate text embeddings
            text_vectors = self.text_embedder.embed_document(document_data)
            self.vector_store.add_text_vectors(text_vectors)
            
            # Generate image embeddings
            image_vectors = self.image_embedder.embed_document_images(document_data)
            self.vector_store.add_image_vectors(image_vectors)
            
            return f"Successfully ingested {file_obj.name} with {len(document_data['pages'])} pages, {len(text_vectors)} text chunks, and {len(image_vectors)} images."
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error ingesting document: {str(e)}"
    
    def query(self, query_text, top_k=5):
        """Query the RAG system with text."""
        try:
            print(f"Received query: {query_text}")
            
            # Check for empty query
            if not query_text or query_text.strip() == "":
                return "Error: Please provide a text query", None
            
            # Text-only mode
            mode = "text"
            image_bytes = None
            
            # Check if we have any vectors
            print(f"Text vectors: {len(self.vector_store.text_vectors)}")
            print(f"Image vectors: {len(self.vector_store.image_vectors)}")
            
            if len(self.vector_store.text_vectors) == 0:
                return "No documents have been indexed yet. Please add documents first.", None
            
            # Retrieve relevant content
            try:
                print("Retrieving relevant content...")
                retrieved_items = self.retriever.retrieve(
                    query_text, 
                    top_k=top_k, 
                    mode=mode, 
                    image_bytes=image_bytes
                )
                print(f"Retrieved {len(retrieved_items)} items")
            except Exception as e:
                print(f"Error during retrieval: {str(e)}")
                return f"Error during retrieval: {str(e)}", None
            
            # Build response
            try:
                print("Building response...")
                response = self.response_builder.build_response(query_text, retrieved_items)
                print("Response built successfully")
            except Exception as e:
                print(f"Error building response: {str(e)}")
                return f"Error building response: {str(e)}", None
            
            # Format the response for display
            formatted_response = response['text_response']
            
            if response['sources']:
                formatted_response += "\n\nSources:\n"
                for source in response['sources']:
                    formatted_response += f"- {source['document_id']}, Page {source['page_num']}\n"
            
            # Return the full response and formatted text
            return formatted_response, response
        
        except Exception as e:
            print(f"Unexpected error in query: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"An unexpected error occurred: {str(e)}", None
    
    def create_ui(self):
        """Create the Gradio UI for the application."""
        with gr.Blocks(title="Aerospace Multimodal RAG") as app:
            gr.Markdown("# Aerospace Multimodal RAG System")
            
            with gr.Row():
                # Left column - Document Ingestion
                with gr.Column(scale=1):
                    gr.Markdown("## Document Ingestion")
                    gr.Markdown("""
                    This system automatically processes PDF files placed in the `data/raw/` directory.
                    
                    You can also upload a PDF document directly below:
                    """)
                    
                    file_input = gr.File(label="Upload Aerospace PDF Document (Optional)")
                    ingest_button = gr.Button("Ingest Document")
                    ingest_output = gr.Textbox(label="Ingestion Status", lines=3)
                    
                    ingest_button.click(
                        fn=self.ingest_document,
                        inputs=[file_input],
                        outputs=[ingest_output]
                    )
                    
                    # Display document count
                    processed_count = len([f for f in os.listdir(self.processed_dir) if f.endswith('.json')])
                    gr.Markdown(f"**{processed_count} documents currently indexed**")
                
                # Right column - Query System
                with gr.Column(scale=2):
                    gr.Markdown("## Query System")
                    
                    query_text = gr.Textbox(label="Text Query", placeholder="Enter your aerospace query here...")
                    query_button = gr.Button("Submit Query", variant="primary")
                    response_text = gr.Textbox(label="Response", lines=20)
                    
                    # Simple query function that returns only a string
                    def simple_query(text):
                        try:
                            result, _ = self.query(text)
                            return result
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            return f"Error: {str(e)}"
                    
                    # Use the simple function directly
                    query_button.click(
                        fn=simple_query,
                        inputs=[query_text],
                        outputs=[response_text]
                    )
            
            return app
    
    def interpreter_mode(self):
        """Run in interpreter mode for direct interaction."""
        print("\n=== Aerospace RAG Interpreter Mode ===")
        print("Type 'exit' or 'quit' to end the session\n")
        
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ['exit', 'quit']:
                print("Exiting interpreter mode.")
                break
            
            response, _ = self.query(query)
            print("\n--- Response ---")
            print(response)
            print("----------------")

# Script execution logic
if __name__ == "__main__":
    # Ensure environment variables are loaded
    print(f"Using environment variables from .env file")
    print(f"OpenAI API Key found: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
    
    # Create the app
    print("Initializing Aerospace RAG system...")
    print("This will check for new PDF files in the data/raw/ directory")
    app = AerospaceRAGApp()
    
    # Display vector store stats
    text_count = len(app.vector_store.text_vectors)
    image_count = len(app.vector_store.image_vectors)
    print(f"Vector store contains {text_count} text vectors and {image_count} image vectors")
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--interpreter':
        # Run in interpreter mode
        app.interpreter_mode()
    else:
        # Run in UI mode
        ui = app.create_ui()
        ui.launch()