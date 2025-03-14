import os
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))


class PDFProcessor:
    def __init__(self, ocr_engine=None, table_extractor=None, formula_parser=None):
        self.ocr_engine = ocr_engine
        self.table_extractor = table_extractor
        self.formula_parser = formula_parser
    
    def process_pdf(self, pdf_path):
        """Process a PDF file and extract text, images, tables and formulas."""
        document = fitz.open(pdf_path)
        
        # Initialize results structure
        result = {
            'metadata': {
                'filename': os.path.basename(pdf_path),
                'pages': len(document),
                'title': document.metadata.get('title', ''),
                'author': document.metadata.get('author', ''),
                'creation_date': document.metadata.get('creationDate', '')
            },
            'pages': []
        }
        
        # Process each page
        for page_idx, page in enumerate(document):
            page_data = {
                'page_num': page_idx + 1,
                'text': page.get_text(),
                'images': [],
                'tables': [],
                'formulas': []
            }
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image for further processing
                image = Image.open(io.BytesIO(image_bytes))
                
                # Use OCR if available and image is large enough to potentially contain text
                if self.ocr_engine and (image.width > 100 and image.height > 100):
                    extracted_text = self.ocr_engine.extract_text(image)
                else:
                    extracted_text = ""
                
                # Store image data
                image_data = {
                    'id': f"page_{page_idx+1}_img_{img_idx+1}",
                    'width': image.width,
                    'height': image.height,
                    'extracted_text': extracted_text,
                    'image_bytes': image_bytes  # Store for embedding generation
                }
                page_data['images'].append(image_data)
            
            # Extract tables if table extractor is available
            if self.table_extractor:
                tables = self.table_extractor.extract_tables(page)
                page_data['tables'] = tables
            
            # Extract formulas if formula parser is available
            if self.formula_parser:
                formulas = self.formula_parser.extract_formulas(page)
                page_data['formulas'] = formulas
            
            result['pages'].append(page_data)
        
        document.close()
        return result