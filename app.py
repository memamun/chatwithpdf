import os
import hashlib
import streamlit as st
from typing import Optional, List, Dict, Tuple
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from autogen import AssistantAgent, UserProxyAgent

# Page configuration
st.set_page_config(
    page_title="TONTRAC AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)



def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_pdf" not in st.session_state:
        st.session_state.processed_pdf = None

def process_pdf(file, api_key: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Optional[FAISS]:
    """Process PDF and create vector store"""
    try:
        with st.spinner("📄 Reading PDF..."):
            with pdfplumber.open(file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            
            if not text.strip():
                st.error("No text could be extracted from the PDF")
                return None

        with st.spinner("🔄 Processing content..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(text)

        with st.spinner("🔍 Creating search index..."):
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vector_store = FAISS.from_texts(chunks, embeddings)
            return vector_store

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def setup_agents(api_key: str):
    """Set up chat agents"""
    config_list = [{
        "model": "gpt-4-1106-preview",
        "api_key": api_key,
    }]
    
    assistant = AssistantAgent(
        name="assistant",
        llm_config={
            "config_list": config_list,
            "temperature": 0.7,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "retrieve_context",
                    "description": "Retrieve relevant context from PDF document",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }]
        },
        system_message="""You are a helpful PDF expert assistant. Follow these guidelines:
        1. Always analyze the retrieved context thoroughly
        2. Provide clear, structured answers based on the document content
        3. If information isn't available, clearly state that
        4. Format responses with markdown for better readability
        5. Be concise and friendly"""
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False
    )
    
    return assistant, user_proxy

def display_chat_message(message: dict):
    """Display a chat message with proper styling"""
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    {message["content"]}
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    {message["content"]}
                </div>
                """, 
                unsafe_allow_html=True
            )

def apply_spacing():
    st.markdown("""
        <style>
        .stContainer {
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def apply_typography_styles():
    st.markdown("""
        <style>
        /* Typography Styles */
        h1 {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E293B; /* Dark color for headings */
        }
        h2 {
            font-size: 2rem;
            font-weight: semi-bold;
            color: #4F46E5; /* Primary color for subheadings */
        }
        h3 {
            font-size: 1.5rem;
            font-weight: semi-bold;
            color: #1E293B;
        }
        p {
            font-size: 1rem;
            line-height: 1.6;
            color: #4B5563; /* Secondary color for body text */
        }

        /* Hide Deploy button */
        .stDeployButton {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

def apply_input_styles():
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            font-size: 1rem;
            padding: 0.75rem;
            border-radius: 0.5rem;
            border: 1px solid #E2E8F0;
        }
        </style>
    """, unsafe_allow_html=True)

def apply_web_fonts():
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
        body {
            font-family: 'Roboto', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

def hide_deploy_button():
    st.markdown("""
        <style>
        /* Hide Deploy button and related elements */
        [data-testid="stAppDeployButton"],
        .stDeployButton,
        [data-testid="stAppDeployButton"] button,
        div[class*="stAppDeployButton"],
        .st-emotion-cache-15wzwg4,
        button[data-testid="stBaseButton-header"] {
            display: none !important;
            visibility: hidden !important;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Configuration")
        
        # File upload
        pdf_file = st.file_uploader(
            "Upload your PDF",
            type="pdf",
            help="Select a PDF file to chat with"
        )
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of text chunks for processing"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between text chunks"
            )
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main content
    st.title("TONTRAC AI Assistant")
    
    if not (pdf_file and api_key):
        st.info("👋 Please upload a PDF and enter your API key to start chatting!")
        return

    # Process PDF if needed
    if pdf_file:
        current_pdf_hash = hashlib.md5(pdf_file.getvalue()).hexdigest()
        if st.session_state.processed_pdf != current_pdf_hash:
            vector_store = process_pdf(pdf_file, api_key, chunk_size, chunk_overlap)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.processed_pdf = current_pdf_hash
                assistant, user_proxy = setup_agents(api_key)
                
                def retrieve_context(query: str) -> str:
                    docs = vector_store.similarity_search(query, k=3)
                    return "\n\n".join([doc.page_content for doc in docs])
                
                user_proxy.register_function(
                    function_map={"retrieve_context": retrieve_context}
                )
                st.session_state.assistant = assistant
                st.session_state.user_proxy = user_proxy

    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        display_chat_message(message)
    st.markdown('</div>', unsafe_allow_html=True)
    

    # Chat input
    if prompt := st.chat_input("Ask about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message({"role": "user", "content": prompt})

        try:
            # Get response
            with st.spinner("Thinking..."):
                response = st.session_state.user_proxy.initiate_chat(
                    st.session_state.assistant,
                    message=prompt,
                    clear_history=False
                )
                
                # Extract the last message from the chat history
                if response and response.chat_history:
                    reply = response.chat_history[-1]['content']
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    display_chat_message({"role": "assistant", "content": reply})
                else:
                    st.error("Failed to get a response. Please try again.")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    apply_web_fonts()
    apply_typography_styles()
    apply_input_styles()
    apply_spacing()
    hide_deploy_button()
    main()