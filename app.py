"""
Task 4: Interactive Chat Interface
Gradio-based web interface for CrediTrust Financial complaint analysis RAG system
"""

import gradio as gr
import json
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')
from task3_rag_pipeline import RAGPipeline

class ComplaintChatInterface:
    """Interactive chat interface for complaint analysis"""
    
    def __init__(self):
        self.rag_pipeline = None
        self.chat_history = []
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialize the RAG pipeline"""
        try:
            print("Loading RAG pipeline...")
            self.rag_pipeline = RAGPipeline(vector_store_path=Path('vector_store'))
            
            if self.rag_pipeline.faiss_index:
                print(f"✅ Pipeline loaded successfully with {self.rag_pipeline.faiss_index.ntotal} complaint chunks")
                return "✅ System ready! Ask me about customer complaints."
            else:
                return "❌ Vector store not found. Please run Tasks 1-2 first to build the complaint database."
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            return f"❌ Error initializing system: {str(e)}"
    
    def process_query(self, question: str, history: list = None) -> tuple:
        """
        Process user question and return response with sources
        
        Args:
            question: User's question
            history: Chat history (for Gradio chatbot)
            
        Returns:
            Tuple of (updated_history, sources_display, sources_json)
        """
        if not question.strip():
            return history, "Please enter a question about customer complaints.", ""
        
        if not self.rag_pipeline or not self.rag_pipeline.faiss_index:
            error_msg = "System not initialized. Please check if vector store exists."
            if history is None:
                history = []
            history.append([question, error_msg])
            return history, error_msg, ""
        
        try:
            # Get RAG response
            response = self.rag_pipeline.run_rag_pipeline(question, k=5)
            
            # Update chat history
            if history is None:
                history = []
            
            history.append([question, response['answer']])
            
            # Format sources for display
            sources_display = self.format_sources_display(response['sources'])
            
            # Prepare sources JSON for download
            sources_json = json.dumps(response['sources'], indent=2, default=str)
            
            # Save interaction
            self.save_interaction(question, response)
            
            return history, sources_display, sources_json
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            if history is None:
                history = []
            history.append([question, error_msg])
            return history, error_msg, ""
    
    def format_sources_display(self, sources: list) -> str:
        """Format sources for user-friendly display"""
        if not sources:
            return "No relevant sources found."
        
        sources_text = "## 📋 Sources Used\\n\\n"
        
        for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
            sources_text += f"**Source {i}** (Similarity: {source['similarity_score']:.3f})\\n"
            sources_text += f"- **Product**: {source['product_category']}\\n"
            sources_text += f"- **Issue**: {source['issue']}\\n"
            sources_text += f"- **Company**: {source.get('company', 'N/A')}\\n"
            sources_text += f"- **Date**: {source.get('date_received', 'N/A')}\\n"
            sources_text += f"- **Text**: {source['chunk_text'][:200]}...\\n\\n"
        
        if len(sources) > 3:
            sources_text += f"*...and {len(sources) - 3} more sources*\\n"
        
        return sources_text
    
    def save_interaction(self, question: str, response: dict):
        """Save interaction for analytics"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': response['answer'],
            'num_sources': response['num_sources'],
            'source_products': list(set([s['product_category'] for s in response['sources']]))
        }
        
        # Save to logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        log_file = logs_dir / f"interactions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(interaction) + '\\n')
    
    def get_sample_questions(self) -> list:
        """Get sample questions for users to try"""
        return [
            "What are the main credit card billing issues?",
            "Why do customers complain about personal loans?", 
            "What problems do people have with money transfers?",
            "What are common savings account complaints?",
            "How do unauthorized charges affect customers?",
            "What loan application issues are reported?",
            "What are the top credit card problems?",
            "Why do money transfers fail?",
            "What account closure issues exist?",
            "How do billing disputes get resolved?"
        ]
    
    def clear_chat(self):
        """Clear chat history"""
        return [], "", ""
    
    def create_interface(self):
        """Create and configure Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto;
        }
        .source-display {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }
        .chat-message {
            margin-bottom: 15px;
        }
        """
        
        with gr.Blocks(css=css, title="CrediTrust Financial - Complaint Analysis AI") as interface:
            
            # Header
            gr.Markdown("""
            # 🏦 CrediTrust Financial - Complaint Analysis AI
            
            **Intelligent complaint analysis powered by RAG (Retrieval-Augmented Generation)**
            
            Ask questions about customer complaints across our products: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.
            """)
            
            # Main chat interface
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Complaint Analysis Chat",
                        height=500,
                        show_label=True,
                        elem_classes=["chat-message"]
                    )
                    
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Ask about customer complaints",
                            placeholder="e.g., What are the main credit card billing issues?",
                            scale=4
                        )
                        submit_btn = gr.Button("🔍 Ask", variant="primary", scale=1)
                        clear_btn = gr.Button("🗑️ Clear", scale=1)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Sample Questions")
                    sample_questions = self.get_sample_questions()
                    for i, question in enumerate(sample_questions[:5]):
                        gr.Button(
                            question, 
                            size="sm",
                            elem_id=f"sample_q_{i}"
                        ).click(
                            fn=lambda q=question: (q, None),
                            outputs=[question_input, chatbot]
                        )
            
            # Sources display
            with gr.Row():
                with gr.Column():
                    sources_display = gr.Markdown(
                        label="📋 Sources & Evidence",
                        elem_classes=["source-display"],
                        visible=True
                    )
            
            # Hidden components for data transfer
            sources_json = gr.Textbox(visible=False)
            
            # Event handlers
            def process_question(question, history):
                return self.process_query(question, history)
            
            # Submit button click
            submit_btn.click(
                fn=process_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, sources_display, sources_json]
            ).then(
                fn=lambda: "",  # Clear input after submit
                outputs=[question_input]
            )
            
            # Enter key press
            question_input.submit(
                fn=process_question,
                inputs=[question_input, chatbot], 
                outputs=[chatbot, sources_display, sources_json]
            ).then(
                fn=lambda: "",
                outputs=[question_input]
            )
            
            # Clear button
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot, sources_display, sources_json]
            )
            
            # Footer
            gr.Markdown("""
            ---
            **About**: This AI system analyzes CFPB complaint data to provide insights into customer issues across financial products.
            
            **Note**: Responses are generated from historical complaint data and should be verified with current business data.
            """)
        
        return interface

def create_streamlit_app():
    """Alternative Streamlit implementation"""
    import streamlit as st
    
    st.set_page_config(
        page_title="CrediTrust Financial - Complaint Analysis AI",
        page_icon="🏦",
        layout="wide"
    )
    
    # Initialize chat interface
    if 'chat_interface' not in st.session_state:
        st.session_state.chat_interface = ComplaintChatInterface()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Header
    st.title("🏦 CrediTrust Financial - Complaint Analysis AI")
    st.markdown("**Intelligent complaint analysis powered by RAG**")
    
    # Sidebar with sample questions
    with st.sidebar:
        st.header("💡 Sample Questions")
        sample_questions = st.session_state.chat_interface.get_sample_questions()
        
        for question in sample_questions[:5]:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.current_question = question
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat")
        
        # Display chat history
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {a}")
            st.markdown("---")
        
        # Input area
        question = st.text_input(
            "Ask about customer complaints:",
            value=getattr(st.session_state, 'current_question', ''),
            key="question_input"
        )
        
        col_submit, col_clear = st.columns([1, 1])
        
        with col_submit:
            if st.button("🔍 Ask", type="primary"):
                if question:
                    history, sources, sources_json = st.session_state.chat_interface.process_query(
                        question, st.session_state.chat_history
                    )
                    st.session_state.chat_history = history
                    st.session_state.current_sources = sources
                    st.rerun()
        
        with col_clear:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.current_sources = ""
                st.rerun()
    
    with col2:
        st.header("📋 Sources")
        if hasattr(st.session_state, 'current_sources'):
            st.markdown(st.session_state.current_sources)

def main():
    """Main application entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CrediTrust Financial Complaint Analysis Interface")
    parser.add_argument("--interface", choices=["gradio", "streamlit"], default="gradio", 
                       help="Choose interface type")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the interface")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    
    args = parser.parse_args()
    
    if args.interface == "gradio":
        # Launch Gradio interface
        chat_interface = ComplaintChatInterface()
        interface = chat_interface.create_interface()
        
        print(f"🚀 Starting Gradio interface on port {args.port}...")
        print(f"🔗 Access the interface at: http://localhost:{args.port}")
        
        interface.launch(
            server_port=args.port,
            share=args.share,
            show_error=True
        )
    
    elif args.interface == "streamlit":
        print("🚀 Starting Streamlit interface...")
        print("Run with: streamlit run app.py")
        create_streamlit_app()

if __name__ == "__main__":
    main()