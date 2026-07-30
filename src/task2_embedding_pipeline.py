"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
Convert cleaned complaint narratives into format suitable for semantic search.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Embedding and vector store imports
from sentence_transformers import SentenceTransformer
import chromadb
import faiss

class TextChunker:
    """Text chunking utility similar to LangChain's RecursiveCharacterTextSplitter"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at word boundary
            if end < len(text):
                # Look for last space within reasonable distance
                last_space = text.rfind(' ', start, end)
                if last_space > start + self.chunk_size * 0.7:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
                
        return chunks

class ComplaintVectorStore:
    """Vector store for complaint embeddings using ChromaDB and FAISS"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self.chroma_collection = None
        
        # Initialize FAISS
        self.faiss_index = None
        self.chunk_metadata = []
        
    def create_stratified_sample(self, df: pd.DataFrame, sample_size: int = 12000) -> pd.DataFrame:
        """Create stratified sample ensuring proportional representation across products"""
        print("Creating stratified sample...")
        
        # Calculate proportional sizes
        product_counts = df['product_category'].value_counts()
        total_records = len(df)
        
        print("Original product distribution:")
        for product, count in product_counts.items():
            print(f"  {product}: {count} ({count/total_records*100:.1f}%)")
        
        sampled_dfs = []
        for product in product_counts.index:
            product_df = df[df['product_category'] == product]
            proportion = product_counts[product] / total_records
            product_sample_size = int(sample_size * proportion)
            
            # Ensure we don't sample more than available
            product_sample_size = min(product_sample_size, len(product_df))
            
            if product_sample_size > 0:
                sample_df = product_df.sample(n=product_sample_size, random_state=42)
                sampled_dfs.append(sample_df)
                print(f"  Sampled {product_sample_size} records from {product}")
        
        sample_df = pd.concat(sampled_dfs, ignore_index=True)
        
        print(f"\nFinal sample size: {len(sample_df)}")
        print("Sample product distribution:")
        sample_counts = sample_df['product_category'].value_counts()
        for product, count in sample_counts.items():
            print(f"  {product}: {count} ({count/len(sample_df)*100:.1f}%)")
            
        return sample_df
    
    def chunk_complaints(self, df: pd.DataFrame, narrative_col: str = 'Consumer complaint narrative') -> List[Dict]:
        """Chunk complaint narratives with metadata"""
        print("Chunking complaint narratives...")
        
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        all_chunks = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing complaints"):
            narrative = row[narrative_col]
            if pd.isna(narrative) or len(str(narrative).strip()) < 20:
                continue
                
            chunks = chunker.split_text(str(narrative))
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_text': chunk,
                    'complaint_id': row.get('Complaint ID', f'complaint_{idx}'),
                    'product_category': row['product_category'],
                    'product': row.get('Product', ''),
                    'issue': row.get('Issue', ''),
                    'sub_issue': row.get('Sub-issue', ''),
                    'company': row.get('Company', ''),
                    'state': row.get('State', ''),
                    'date_received': str(row.get('Date received', '')),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
                all_chunks.append(chunk_data)
        
        print(f"Generated {len(all_chunks)} chunks from {len(df)} complaints")
        
        # Calculate chunk size statistics
        chunk_lengths = [len(chunk['chunk_text']) for chunk in all_chunks]
        print(f"Chunk length statistics:")
        print(f"  Mean: {np.mean(chunk_lengths):.1f} chars")
        print(f"  Median: {np.median(chunk_lengths):.1f} chars")
        print(f"  Min: {np.min(chunk_lengths)} chars")
        print(f"  Max: {np.max(chunk_lengths)} chars")
        
        return all_chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for all chunks"""
        print("Generating embeddings...")
        
        texts = [chunk['chunk_text'] for chunk in chunks]
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(embeddings)
        print(f"Generated embeddings shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def create_chromadb_store(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        """Create ChromaDB vector store"""
        print("Creating ChromaDB vector store...")
        
        # Create or get collection
        try:
            self.chroma_collection = self.chroma_client.get_collection("complaint_embeddings")
            self.chroma_client.delete_collection("complaint_embeddings")
        except:
            pass
            
        self.chroma_collection = self.chroma_client.create_collection(
            name="complaint_embeddings",
            metadata={"description": "CFPB complaint embeddings for RAG system"}
        )
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk['chunk_text'] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {k: v for k, v in chunk.items() if k != 'chunk_text'}
            # Convert all values to strings for ChromaDB compatibility
            metadata = {k: str(v) for k, v in metadata.items()}
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 1000
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to ChromaDB"):
            end_idx = min(i + batch_size, len(chunks))
            
            self.chroma_collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print(f"Added {len(chunks)} chunks to ChromaDB")
    
    def create_faiss_store(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """Create FAISS vector store"""
        print("Creating FAISS vector store...")
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store metadata separately
        self.chunk_metadata = chunks.copy()
        
        print(f"Added {self.faiss_index.ntotal} vectors to FAISS index")
    
    def save_vector_stores(self, output_dir: Path) -> None:
        """Save both ChromaDB and FAISS stores to disk"""
        print("Saving vector stores...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ChromaDB (persist to disk)
        chroma_path = output_dir / "chromadb"
        if self.chroma_collection:
            # ChromaDB persistence requires different approach
            print(f"ChromaDB collection created (in-memory)")
            
            # Save collection metadata
            collection_info = {
                'name': self.chroma_collection.name,
                'count': self.chroma_collection.count(),
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }
            
            with open(output_dir / "chromadb_info.json", 'w') as f:
                json.dump(collection_info, f, indent=2)
        
        # Save FAISS index
        if self.faiss_index:
            faiss_path = output_dir / "faiss_index.index"
            faiss.write_index(self.faiss_index, str(faiss_path))
            
            # Save metadata
            metadata_path = output_dir / "faiss_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            
            print(f"FAISS index saved to {faiss_path}")
            print(f"Metadata saved to {metadata_path}")
        
        # Save model info
        model_info = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'total_chunks': len(self.chunk_metadata) if self.chunk_metadata else 0
        }
        
        with open(output_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("Vector stores saved successfully!")
    
    def load_faiss_store(self, vector_store_path: Path):
        """Load FAISS index and metadata"""
        faiss_path = vector_store_path / "faiss_index.index"
        metadata_path = vector_store_path / "faiss_metadata.pkl"
        
        if faiss_path.exists() and metadata_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            with open(metadata_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            
            print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return True
        return False
    
    def search_similar(self, query: str, k: int = 5, use_faiss: bool = True) -> List[Dict]:
        """Search for similar chunks using query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        if use_faiss and self.faiss_index:
            # Normalize query for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunk_metadata):
                    result = self.chunk_metadata[idx].copy()
                    result['similarity_score'] = float(score)
                    results.append(result)
            
            return results
        
        elif self.chroma_collection:
            # Search ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding[0].tolist()],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = results['metadatas'][0][i].copy()
                result['chunk_text'] = results['documents'][0][i]
                result['similarity_score'] = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                formatted_results.append(result)
            
            return formatted_results
        
        else:
            print("No vector store loaded!")
            return []

def main():
    """Main execution function for Task 2"""
    print("=" * 60)
    print("Task 2: Text Chunking, Embedding, and Vector Store Indexing")
    print("=" * 60)
    
    # Setup paths
    data_dir = Path('../data')
    processed_dir = data_dir / 'processed'
    vector_store_dir = Path('../vector_store')
    
    try:
        # Load cleaned data from Task 1
        data_path = processed_dir / 'filtered_complaints.csv'
        if not data_path.exists():
            print(f"Error: {data_path} not found. Please run Task 1 first.")
            return
        
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} cleaned complaints")
        
        # Initialize vector store
        vector_store = ComplaintVectorStore()
        
        # Create stratified sample
        sample_df = vector_store.create_stratified_sample(df, sample_size=12000)
        
        # Chunk complaints
        chunks = vector_store.chunk_complaints(sample_df)
        
        # Generate embeddings
        embeddings = vector_store.generate_embeddings(chunks)
        
        # Create vector stores
        vector_store.create_chromadb_store(chunks, embeddings)
        vector_store.create_faiss_store(embeddings, chunks)
        
        # Save to disk
        vector_store.save_vector_stores(vector_store_dir)
        
        # Test search functionality
        print("\n" + "="*40)
        print("Testing search functionality:")
        print("="*40)
        
        test_queries = [
            "billing error on credit card",
            "loan application denied",
            "money transfer failed"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = vector_store.search_similar(query, k=3, use_faiss=True)
            for i, result in enumerate(results[:2], 1):
                print(f"  {i}. [{result['product_category']}] Score: {result['similarity_score']:.3f}")
                print(f"     {result['chunk_text'][:100]}...")
        
        print("\n" + "=" * 60)
        print("Task 2 completed successfully!")
        print("=" * 60)
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Processed {len(sample_df)} complaints")
        print(f"- Generated {len(chunks)} text chunks") 
        print(f"- Created {embeddings.shape[0]} embeddings")
        print(f"- Vector stores saved to {vector_store_dir}")
        
    except Exception as e:
        print(f"Error in Task 2: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()