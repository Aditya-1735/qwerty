from sentence_transformers import SentenceTransformer
from typing import List
import torch
import os
from langchain_core.embeddings import Embeddings

class OptimizedEmbeddings(Embeddings):
    """
    Optimized embedding class for fast processing of large datasets.
    Compatible with LangChain and Qdrant.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-distilroberta-v1"):
        """
        Initialize with SentenceTransformer model
        
        Args:
            model_name: HuggingFace model name. Options:
                - "sentence-transformers/all-distilroberta-v1" (768 dim, fast, good quality)
                - "sentence-transformers/all-MiniLM-L6-v2" (384 dim, smaller, decent quality)
                - "sentence-transformers/all-mpnet-base-v2" (768 dim, slower, best quality)
        """
        print(f"🔄 Loading embedding model: {model_name}")
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name
        
        # Optimize for inference
        self.model.eval()
        
        # Performance settings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
        
        print(f"✅ Model loaded on: {self.device}")
        if self.device == 'cuda':
            print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embed multiple documents efficiently
        
        Args:
            texts: List of text documents to embed
            batch_size: Batch size for processing (adjust based on GPU memory)
        
        Returns:
            List of embeddings as lists of floats
        """
        if not texts:
            return []
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            # Show progress for large batches
            if len(texts) > 100 and batch_num % 10 == 0:
                print(f"   📊 Processing batch {batch_num}/{total_batches}")
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=batch_size,
                    normalize_embeddings=True  # Better for similarity search
                )
            
            all_embeddings.extend(embeddings.tolist())
            
            # Clear GPU cache periodically
            if self.device == 'cuda' and batch_num % 20 == 0:
                torch.cuda.empty_cache()
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Query text to embed
        
        Returns:
            Embedding as list of floats
        """
        with torch.no_grad():
            embedding = self.model.encode(
                [text], 
                convert_to_tensor=False,
                normalize_embeddings=True
            )
        return embedding[0].tolist()
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        """Make class callable for compatibility"""
        return self.embed_documents(input_texts)

def get_embedding_function():
    """
    Get the optimized embedding function for RAG system.
    
    Returns:
        OptimizedEmbeddings instance ready for use with LangChain/Qdrant
    """
    # Choose model based on your needs:
    # - Fast processing: "sentence-transformers/all-distilroberta-v1"
    # - Smaller size: "sentence-transformers/all-MiniLM-L6-v2" 
    # - Best quality: "sentence-transformers/all-mpnet-base-v2"
    
    model_name = "sentence-transformers/all-distilroberta-v1"
    return OptimizedEmbeddings(model_name=model_name)

# Test function
def test_embedding_function():
    """Test the embedding function"""
    print("🧪 Testing embedding function...")
    
    embeddings = get_embedding_function()
    
    # Test single query
    test_query = "What are the symptoms of diabetes?"
    query_embedding = embeddings.embed_query(test_query)
    
    # Test multiple documents
    test_docs = [
        "Diabetes is a chronic condition affecting blood sugar levels.",
        "Common symptoms include increased thirst and frequent urination.",
        "Treatment often involves medication and lifestyle changes."
    ]
    doc_embeddings = embeddings.embed_documents(test_docs)
    
    print(f"✅ Query embedding dimension: {len(query_embedding)}")
    print(f"✅ Document embeddings count: {len(doc_embeddings)}")
    print(f"✅ Model dimension: {embeddings.get_dimension()}")
    print("🎉 Embedding function test passed!")
    
    return True

if __name__ == "__main__":
    test_embedding_function()