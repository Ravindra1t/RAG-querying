Enhanced RAG API
This project provides a robust and scalable Retrieval-Augmented Generation (RAG) API built with FastAPI. It's designed to answer questions about documents by intelligently scraping, processing, and retrieving relevant information. The system supports various document types and uses an optimized hybrid retrieval approach to ensure accuracy and speed.

Features ✨
Versatile Document Support: Process content from websites (with a smart fallback to Selenium for dynamic content), PDF files, and DOCX documents.

Smart Hybrid Retrieval: Combines semantic search (using Sentence-Transformers and FAISS) with keyword-based search (using BM25) for highly accurate context retrieval.

Optimized LLM Generation: Batches multiple questions into a single API call to the Groq API, significantly reducing latency and cost.

Embedding Caching: Caches document embeddings to avoid redundant computation, speeding up subsequent queries on the same document.

Scalable Architecture: Built on FastAPI for high-performance and asynchronous request handling.

API Endpoints 🚀
/v1/api
This is the main endpoint for submitting questions and receiving answers based on a provided document.

Method: POST

Description: Processes a document (URL) and answers a list of questions based on its content.

Request Body:

JSON

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of the document?",
    "What are the key points mentioned?"
  ]
}
Example Response:

JSON

{
  "answers": [
    "1) The main topic of the document is...",
    "2) The key points mentioned are..."
  ]
}
/
Method: GET

Description: Provides a simple overview of the API's features and available endpoints.

How to Run the Server 💻
Clone the repository (if applicable) or save the Python script as a file (e.g., main.py).

Install dependencies:
This project requires several libraries. You can install them by running the following command:

Bash

pip install fastapi uvicorn requests sentence-transformers faiss-cpu rank-bm25 PyMuPDF python-docx beautifulsoup4 selenium pyngrok nest-asyncio nltk
Note: You may also need to install a web browser (like Chrome or Chromium) for Selenium to work properly.

Configure API Keys:
Open the script and replace the placeholder "keep your key here" values in the Config class with your actual Groq API key and ngrok authtoken.

Groq API Key: Get it from Groq Console.

ngrok Authtoken: Get it from ngrok Dashboard.

Run the application:
Execute the script from your terminal. This will start the FastAPI server and create an ngrok tunnel.

Bash

python main.py
The public URL for your API will be printed to the console.

Code Structure 🛠️
Config: Manages all global settings, including API keys, chunk sizes, and scraping parameters.

EmbeddingCache: Implements a simple file-based cache to store embeddings, preventing redundant API calls for previously processed documents.

WebScraper: Handles web content extraction, leveraging both requests/BeautifulSoup for static content and Selenium for dynamic, JavaScript-heavy pages.

EnhancedDocumentLoader: Acts as a central manager for loading content from different sources (PDF, DOCX, web).

TextProcessor: Cleans raw text and chunks it into smaller, overlapping segments, preparing it for the retrieval step.

SmartRetriever: The heart of the RAG system. It builds a hybrid index combining FAISS (for semantic search) and BM25 (for keyword search) to find the most relevant document chunks.

AnswerGenerator: Optimizes communication with the LLM. It's responsible for batching multiple questions into a single, efficient API call and parsing the structured response.

EnhancedRAGSystem: Orchestrates the entire process—loading, chunking, indexing, retrieving, and generating answers.

FastAPI Endpoints: Defines the API interface, handling incoming requests and returning JSON responses.
