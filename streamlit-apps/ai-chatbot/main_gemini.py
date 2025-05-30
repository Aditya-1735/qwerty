import streamlit as st
import os
import warnings

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time

# Delayed imports to avoid conflicts
def load_dependencies():
    from dotenv import load_dotenv
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from get_embedding_function import get_embedding_function
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.runnables import RunnablePassthrough
    
    load_dotenv()
    return {
        'QdrantVectorStore': QdrantVectorStore,
        'QdrantClient': QdrantClient,
        'get_embedding_function': get_embedding_function,
        'PromptTemplate': PromptTemplate,
        'ChatGoogleGenerativeAI': ChatGoogleGenerativeAI,
        'create_stuff_documents_chain': create_stuff_documents_chain,
        'RunnablePassthrough': RunnablePassthrough
    }

# Load environment variables
deps = load_dependencies()

# Qdrant configuration
QDRANT_COLLECTION_NAME = "health_documents"  # Your collection name

# Page configuration
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2E86AB;
    }
    .user-message {
        background-color: #E8F4FD;
        border-left-color: #2E86AB;
    }
    .assistant-message {
        background-color: #F0F8F0;
        border-left-color: #28A745;
    }
    .sidebar-info {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #DEE2E6;
    }
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot with Qdrant Cloud"""
    try:
        # Load environment variables
        google_api_key = os.getenv("GOOGLE_API_KEY")
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not google_api_key:
            st.error("❌ GOOGLE_API_KEY not found in environment variables")
            st.stop()
        
        if not qdrant_url or not qdrant_api_key:
            st.error("❌ QDRANT_URL or QDRANT_API_KEY not found in environment variables")
            st.stop()
        
        # Initialize Qdrant client
        qdrant_client = deps['QdrantClient'](
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Initialize embeddings
        embeddings = deps['get_embedding_function']()
        
        # Create Qdrant vector store
        vectorstore = deps['QdrantVectorStore'](
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Get top 5 most relevant documents
        )
        
        # Initialize the LLM
        llm = deps['ChatGoogleGenerativeAI'](
            model="gemini-2.5-pro-preview-05-06",
            google_api_key=google_api_key,
            temperature=0.3
        )
        
        # Create specialized health prompt
        prompt = deps['PromptTemplate'].from_template("""
        You are a knowledgeable AI Health Assistant. Use the provided medical context to answer health-related questions accurately and helpfully.
        
        IMPORTANT GUIDELINES:
        - Provide evidence-based information from the context
        - Always emphasize that this is for informational purposes only
        - Recommend consulting healthcare professionals for medical advice
        - If the context doesn't contain relevant information, clearly state this
        - Be empathetic and supportive in your responses
        - Use clear, understandable language
        
        Question: {question}
        
        Medical Context:
        {context}
        
        Health Assistant Response:
        """)
        
        # Create the document chain
        document_chain = deps['create_stuff_documents_chain'](llm, prompt)
        
        # Create retrieval chain
        retrieval_chain = {
            "context": retriever,
            "question": deps['RunnablePassthrough']()
        } | document_chain
        
        return retrieval_chain, vectorstore, qdrant_client
    
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {str(e)}")
        st.stop()

def display_chat_message(role, content):
    """Display a chat message with proper styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🏥 Health Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Health Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Information")
        st.markdown("""
        <div class="sidebar-info">
            <h4>🎯 What I can help with:</h4>
            <ul>
                <li>General health information</li>
                <li>Symptoms and conditions</li>
                <li>Medical terminology</li>
                <li>Treatment options</li>
                <li>Preventive care</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Database info
        if st.button("📊 Database Status"):
            try:
                _, vectorstore, qdrant_client = initialize_chatbot()
                
                # Get collection info
                collection_info = qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
                
                st.success(f"✅ Qdrant Cloud connected successfully!")
                st.info(f"📄 Available documents: {collection_info.points_count}")
                st.info(f"🔗 Collection: {QDRANT_COLLECTION_NAME}")
                st.info(f"🔍 Search method: Semantic similarity")
                
            except Exception as e:
                st.error(f"❌ Database connection error: {str(e)}")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Warning disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong><br>
        This AI assistant provides general health information for educational purposes only. 
        Always consult with qualified healthcare professionals for medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = """
        Hello! I'm your AI Health Assistant. I can help answer your health-related questions using medical knowledge from my database.
        
        Feel free to ask me about:
        - Symptoms and conditions
        - Medical terminology
        - Treatment options
        - Preventive care
        - General health information
        
        How can I assist you today?
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your health concerns..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Generate response
        with st.spinner("🔍 Searching medical knowledge base..."):
            try:
                retrieval_chain, _, _ = initialize_chatbot()
                
                # Get response
                response = retrieval_chain.invoke(prompt)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                display_chat_message("assistant", response)
                
            except Exception as e:
                error_msg = f"❌ I apologize, but I encountered an error while processing your question: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                display_chat_message("assistant", error_msg)
        
        # Rerun to update the display
        st.rerun()

if __name__ == "__main__":
    main()