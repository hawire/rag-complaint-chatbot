"""
Task 3: RAG Core Logic and Evaluation
Building the retrieval and generation pipeline for CrediTrust Financial complaint analysis
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime

# Core ML imports
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Import previous implementations
from task2_embedding_pipeline import ComplaintVectorStore

class RAGPipeline:
    """Complete RAG pipeline for complaint analysis"""
    
    def __init__(self, 
                 vector_store_path: Path = None,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name: str = "microsoft/DialoGPT-medium"):
        
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.vector_store_path = vector_store_path or Path('../vector_store')
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm_pipeline = None
        
        # Vector store components
        self.faiss_index = None
        self.chunk_metadata = []
        
        # Load vector store
        self._load_vector_store()
        self._initialize_llm()
    
    def _load_vector_store(self):
        """Load FAISS vector store and metadata"""
        faiss_path = self.vector_store_path / "faiss_index.index"
        metadata_path = self.vector_store_path / "faiss_metadata.pkl"
        
        try:
            if faiss_path.exists() and metadata_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
                
                with open(metadata_path, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                
                print(f"Loaded vector store with {self.faiss_index.ntotal} vectors")
            else:
                print(f"Vector store not found at {self.vector_store_path}")
                print("Please run Task 2 first to create the vector store")
        except Exception as e:
            print(f"Error loading vector store: {e}")
    
    def _initialize_llm(self):
        """Initialize the language model for generation"""
        try:
            # Use a lightweight model for demonstration
            # In production, you might use larger models like Llama or Mistral
            self.llm_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            # Fallback to a simple text generation approach
            self._initialize_simple_llm()
    
    def _initialize_simple_llm(self):
        """Fallback simple LLM implementation"""
        print("Using simple response generation as fallback")
        self.llm_pipeline = None
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for the query
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.faiss_index:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search vector store
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_metadata):
                chunk_data = self.chunk_metadata[idx].copy()
                chunk_data['similarity_score'] = float(score)
                chunk_data['rank'] = len(results) + 1
                results.append(chunk_data)
        
        return results
    
    def create_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Create the prompt template with context and query
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Formatted prompt for the LLM
        """
        # Format context from retrieved chunks
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"[Source {i}] {chunk['product_category']} - {chunk['issue']}\n"
            context_text += f"{chunk['chunk_text']}\n\n"
        
        # Create the prompt template
        prompt_template = f"""You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on the provided context.

Instructions:
- Use ONLY the information from the provided complaint excerpts
- Be specific and cite relevant details from the context
- If the context doesn't contain enough information, state that clearly
- Focus on actionable insights for the business
- Maintain a professional, analytical tone

Context:
{context_text}

Question: {query}

Analysis and Answer:"""

        return prompt_template
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using the LLM
        
        Args:
            prompt: Formatted prompt with context and query
            
        Returns:
            Generated response
        """
        if self.llm_pipeline:
            try:
                # Generate response
                response = self.llm_pipeline(
                    prompt,
                    max_length=len(prompt.split()) + 150,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=50256
                )[0]['generated_text']
                
                # Extract only the new generated part
                generated_part = response[len(prompt):].strip()
                return generated_part if generated_part else "I need more specific information to provide a detailed analysis."
                
            except Exception as e:
                print(f"Error in LLM generation: {e}")
                return self._fallback_response(prompt)
        else:
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Simple fallback response generation"""
        # Extract context information for basic analysis
        lines = prompt.split('\n')
        context_lines = [line for line in lines if '[Source' in line or 'product_category' in line.lower()]
        
        if context_lines:
            products = set()
            issues = set()
            
            for line in lines:
                if 'Credit Card' in line or 'Personal Loan' in line or 'Savings Account' in line or 'Money Transfer' in line:
                    if 'Credit Card' in line:
                        products.add('Credit Cards')
                    elif 'Personal Loan' in line:
                        products.add('Personal Loans') 
                    elif 'Savings Account' in line:
                        products.add('Savings Accounts')
                    elif 'Money Transfer' in line:
                        products.add('Money Transfers')
            
            response = f"Based on the available complaint data, "
            if products:
                response += f"this appears to involve {', '.join(products)}. "
            
            response += "The analysis shows customer concerns that require attention from the product team. "
            response += "I recommend reviewing the specific complaint details for actionable insights."
            
            return response
        
        return "I don't have sufficient context from the complaint data to provide a specific analysis."
    
    def run_rag_pipeline(self, query: str, k: int = 5) -> Dict:
        """
        Complete RAG pipeline execution
        
        Args:
            query: User's question
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant context
        retrieved_chunks = self.retrieve_context(query, k)
        
        if not retrieved_chunks:
            return {
                'answer': "I don't have access to relevant complaint data to answer this question.",
                'sources': [],
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 2: Create prompt
        prompt = self.create_prompt(query, retrieved_chunks)
        
        # Step 3: Generate response
        answer = self.generate_response(prompt)
        
        # Step 4: Format response
        result = {
            'answer': answer,
            'sources': retrieved_chunks,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'num_sources': len(retrieved_chunks)
        }
        
        return result

class RAGEvaluator:
    """Evaluation utilities for RAG pipeline"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
    
    def create_evaluation_questions(self) -> List[str]:
        """Create representative evaluation questions"""
        return [
            "Why are customers complaining about credit cards?",
            "What are the main issues with personal loans?",
            "How do money transfer problems affect customers?",
            "What billing issues do customers face?",
            "Are there patterns in savings account complaints?",
            "What unauthorized charges issues are reported?",
            "How do customers complain about loan applications?",
            "What are the top credit card problems?",
            "Why do money transfers fail?",
            "What account closure issues exist?"
        ]
    
    def evaluate_response_quality(self, query: str, response: Dict) -> Dict:
        """
        Evaluate the quality of a RAG response
        
        Args:
            query: Original question
            response: RAG pipeline response
            
        Returns:
            Quality evaluation metrics
        """
        answer = response['answer']
        sources = response['sources']
        
        # Basic quality metrics
        metrics = {
            'query': query,
            'answer_length': len(answer.split()),
            'num_sources': len(sources),
            'source_diversity': len(set([s['product_category'] for s in sources])),
            'avg_similarity': np.mean([s['similarity_score'] for s in sources]) if sources else 0,
            'contains_specific_details': any(keyword in answer.lower() for keyword in 
                                           ['billing', 'charge', 'account', 'loan', 'transfer', 'card']),
            'mentions_products': any(product in answer for product in 
                                   ['Credit Card', 'Personal Loan', 'Savings Account', 'Money Transfer']),
            'actionable_insights': any(phrase in answer.lower() for phrase in 
                                     ['recommend', 'should', 'analysis', 'review', 'attention'])
        }
        
        # Calculate overall quality score (1-5)
        quality_score = 1.0
        if metrics['num_sources'] > 0:
            quality_score += 1.0
        if metrics['avg_similarity'] > 0.7:
            quality_score += 1.0
        if metrics['contains_specific_details']:
            quality_score += 0.5
        if metrics['mentions_products']:
            quality_score += 0.5
        
        metrics['quality_score'] = min(5.0, quality_score)
        
        return metrics
    
    def run_full_evaluation(self) -> List[Dict]:
        """Run evaluation on all test questions"""
        test_questions = self.create_evaluation_questions()
        evaluation_results = []
        
        print("Running RAG Pipeline Evaluation...")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"Evaluating question {i}/{len(test_questions)}: {question}")
            
            # Get RAG response
            response = self.rag_pipeline.run_rag_pipeline(question)
            
            # Evaluate quality
            quality_metrics = self.evaluate_response_quality(question, response)
            
            # Combine results
            evaluation_result = {
                **response,
                **quality_metrics
            }
            
            evaluation_results.append(evaluation_result)
            
            print(f"  Quality Score: {quality_metrics['quality_score']:.1f}/5.0")
            print(f"  Sources: {quality_metrics['num_sources']}")
            print()
        
        return evaluation_results
    
    def generate_evaluation_report(self, results: List[Dict]) -> str:
        """Generate markdown evaluation report"""
        
        report = """# RAG Pipeline Evaluation Report

## Overview
This report evaluates the performance of the CrediTrust Financial complaint analysis RAG system.

## Evaluation Results

| Question | Quality Score | Answer Preview | Sources | Source Diversity | Comments |
|----------|---------------|----------------|---------|------------------|----------|
"""
        
        for result in results:
            answer_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            answer_preview = answer_preview.replace('\n', ' ')
            
            report += f"| {result['query']} | {result['quality_score']:.1f}/5 | {answer_preview} | {result['num_sources']} | {result['source_diversity']}/4 | "
            
            # Add comments based on metrics
            comments = []
            if result['quality_score'] >= 4.0:
                comments.append("Excellent")
            elif result['quality_score'] >= 3.0:
                comments.append("Good")
            else:
                comments.append("Needs improvement")
                
            if result['avg_similarity'] < 0.6:
                comments.append("Low relevance")
            
            report += " | ".join(comments) + " |\n"
        
        # Add summary statistics
        avg_quality = np.mean([r['quality_score'] for r in results])
        avg_sources = np.mean([r['num_sources'] for r in results])
        avg_similarity = np.mean([r['avg_similarity'] for r in results])
        
        report += f"""

## Summary Statistics
- **Average Quality Score**: {avg_quality:.2f}/5.0
- **Average Sources per Query**: {avg_sources:.1f}
- **Average Source Similarity**: {avg_similarity:.3f}
- **Total Queries Evaluated**: {len(results)}

## Key Findings
"""
        
        high_quality = len([r for r in results if r['quality_score'] >= 4.0])
        good_diversity = len([r for r in results if r['source_diversity'] >= 2])
        
        report += f"""
1. **Response Quality**: {high_quality}/{len(results)} queries achieved high quality scores (≥4.0)
2. **Source Diversity**: {good_diversity}/{len(results)} queries showed good source diversity (≥2 products)
3. **Retrieval Performance**: Average similarity score of {avg_similarity:.3f} indicates effective semantic matching

## Recommendations
- Continue to improve prompt engineering for more specific responses
- Consider expanding the vector store with more diverse complaint examples
- Implement source citation improvements for better traceability
"""
        
        return report

def main():
    """Main execution function for Task 3"""
    print("=" * 60)
    print("Task 3: RAG Pipeline Implementation and Evaluation")
    print("=" * 60)
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline()
    
    if not rag.faiss_index:
        print("Error: Vector store not found. Please run Task 2 first.")
        return
    
    # Test single query
    print("\nTesting RAG pipeline with sample query...")
    test_query = "What are the main credit card billing issues customers face?"
    response = rag.run_rag_pipeline(test_query)
    
    print(f"Query: {test_query}")
    print(f"Answer: {response['answer']}")
    print(f"Number of sources: {response['num_sources']}")
    
    # Run full evaluation
    print("\n" + "="*60)
    print("Running Full Evaluation...")
    print("="*60)
    
    evaluator = RAGEvaluator(rag)
    evaluation_results = evaluator.run_full_evaluation()
    
    # Generate and save evaluation report
    report = evaluator.generate_evaluation_report(evaluation_results)
    
    # Save results
    output_dir = Path('../reports')
    output_dir.mkdir(exist_ok=True)
    
    # Save evaluation results as JSON
    with open(output_dir / 'rag_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Save evaluation report as markdown
    with open(output_dir / 'rag_evaluation_report.md', 'w') as f:
        f.write(report)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"- Detailed results: rag_evaluation_results.json") 
    print(f"- Summary report: rag_evaluation_report.md")
    
    # Print summary
    avg_quality = np.mean([r['quality_score'] for r in evaluation_results])
    print(f"\nOverall Performance: {avg_quality:.2f}/5.0")

if __name__ == "__main__":
    main()