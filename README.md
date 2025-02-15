Document Processing Pipeline with Vector Search Reranking

Overview

This Streamlit application allows users to upload a PDF document, extract its text, and query the content using a vector search-based retrieval system. The pipeline includes document chunking, vector embedding, and query-based reranking with citations.

Installation

Clone the repository:

git clone https://github.com/Pragadishwaran01/Document-Processing-Pipeline-with-Vector-Search-Reranking

cd Document-Processing-Pipeline-with-Vector-Search-Reranking

Install dependencies:

pip install -r requirements.txt

Set up OpenAI API Key:

Create a .env file in the root directory and add the following:

OPENAI_API_KEY=<your_openai_api_key>

Usage:

Run the Streamlit application:

streamlit run app.py

Upload a PDF file.

Ask questions about the document content.

View the retrieved answer along with citations and source details.

File Structure:

📂 project_root

├── app.py                # Main Streamlit application

├── requirements.txt      # Dependencies

├── .env                  # Environment variables (not included in repo)

├── README.md             # Documentation
