import pytesseract
from PIL import Image

class OCREngine:
    def __init__(self, tesseract_cmd=None, lang='eng'):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang
    
    def extract_text(self, image):
        """Extract text from an image using OCR."""
        try:
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""