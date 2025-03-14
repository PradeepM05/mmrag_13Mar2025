# Aerospace Multimodal RAG System

A Retrieval-Augmented Generation (RAG) system specialized for aerospace documents, capable of processing both text and images from PDF documents.

## Features

- PDF document ingestion with OCR capabilities
- Text and image embedding generation
- Multimodal retrieval (text and image queries)
- Integration with LLM for response generation
- User-friendly Gradio interface

## Project Structure

```
aerospace-rag/
├── data/                  # Data storage
│   ├── raw/               # Raw PDF documents
│   ├── processed/         # Processed document data
│   └── embeddings/        # Vector embeddings
├── src/                   # Source code
│   ├── ingestion/         # Document processing modules
│   ├── embedding/         # Text and image embedding modules
│   ├── retrieval/         # Vector storage and retrieval
│   ├── generation/        # LLM integration and response building
│   └── utils/             # Utility functions
├── ui/                    # User interface
│   └── app.py             # Gradio web application
├── tests/                 # Test cases
└── notebooks/             # Jupyter notebooks for exploration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PradeepM05/mmrag_13Mar2025.git
cd mmrag_13Mar2025
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Place aerospace PDF documents in the `data/raw/` directory.

2. Run the application:
```bash
python ui/app.py
```

3. Access the web interface at http://127.0.0.1:7860

## License

MIT 