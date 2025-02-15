import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import tempfile

load_dotenv()

# Set OpenAI API key from st.secrets
if "openai" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None

st.title("Document Processing Pipeline with Vector Search Reranking")
st.write("Upload a PDF document, then ask questions to get answers with citations.")

# Only show API key input if not provided in secrets
if "openai" not in st.secrets:
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file and return text with page numbers."""
    text_by_page = {}
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text_by_page[i] = page.extract_text() or ""
    return text_by_page

def create_documents_with_metadata(text_by_page, pdf_name):
    """Create Document objects with page number and source metadata."""
    documents = []
    
    for page_num, text in text_by_page.items():
        if not text.strip(): 
            continue

        metadata = {
            "source": pdf_name,
            "page": page_num,
            "citation": f"{pdf_name}, page {page_num}"
        }

        doc = Document(text=text, metadata=metadata)
        documents.append(doc)
    
    return documents

def process_pdf(uploaded_file):
    """Process the uploaded PDF and create a query engine."""
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please provide your OpenAI API Key in the app's secrets or enter it above.")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    
    try:
        # Step 1: Extract text with page numbers
        text_by_page = extract_text_from_pdf(temp_path)
        
        # Step 2: Create documents with metadata
        documents = create_documents_with_metadata(text_by_page, uploaded_file.name)
        
        if not documents:
            st.error("Could not extract any text from the PDF. Please try another file.")
            return None
        
        # Step 3: Chunk text semantically
        embed_model = OpenAIEmbedding()
        parser = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=95,
        )
        nodes = parser.get_nodes_from_documents(documents)
        
        # Step 4: Setup vector store
        client = qdrant_client.QdrantClient(location=":memory:")
        vector_store = QdrantVectorStore(client=client, collection_name="document_qa")
        
        # Step 5: Create index
        index = VectorStoreIndex(
            nodes, 
            vector_store=vector_store, 
            embed_model=embed_model,
            show_progress=True
        )

        retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
        
        text_qa_template = PromptTemplate(
            """
            Context information is below. Given the context information and not prior knowledge, 
            answer the query. Include inline citations like [1], [2], etc. that refer to the source documents.
            
            Query: {query_str}
            
            Context: {context_str}
            
            Answer: 
            """
        )
        
        refine_template = PromptTemplate(
            """
            The original query is: {query_str}
            
            We have provided an existing answer: {existing_answer}
            
            We have the opportunity to refine the existing answer with some more context below.
            (Only if needed) Provide inline citations [1], [2], etc. for any new information.
            
            Context: {context_msg}
            
            Refined Answer: 
            """
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode="refine",
            text_qa_template=text_qa_template,
            refine_template=refine_template
        )
        
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.80)
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[postprocessor]
        )
        
        return query_engine
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner("Processing document..."):
        st.session_state.query_engine = process_pdf(uploaded_file)
        if st.session_state.query_engine:
            st.session_state.pdf_processed = True
            st.session_state.pdf_name = uploaded_file.name
            st.success(f"Successfully processed '{uploaded_file.name}'")

if st.session_state.pdf_processed:
    if st.button("Process a different PDF"):
        st.session_state.query_engine = None
        st.session_state.pdf_processed = False
        st.session_state.pdf_name = None
        st.experimental_rerun()

if st.session_state.pdf_processed and st.session_state.query_engine:
    st.write(f"Currently using: {st.session_state.pdf_name}")
    query = st.text_input("Ask a question about the document:")
    
    if query:
        with st.spinner("Generating answer..."):
            response = st.session_state.query_engine.query(query)

            st.markdown("### Answer")
            st.write(str(response))

            if hasattr(response, 'source_nodes') and response.source_nodes:
                st.markdown("### Citations")
                for i, node in enumerate(response.source_nodes, 1):
                    citation = node.node.metadata.get('citation', 'No citation available')
                    st.write(f"[{i}] {citation}")

                with st.expander("Show detailed sources"):
                    for i, node in enumerate(response.source_nodes, 1):
                        st.markdown(f"**Source {i}**")
                        st.write(f"Score: {node.score:.4f}")
                        st.write(f"Citation: {node.node.metadata.get('citation', 'No citation available')}")
                        st.text(node.node.text)
                        st.markdown("---")