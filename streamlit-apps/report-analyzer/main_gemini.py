import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from typing import List, Dict
import time
import hashlib

# Configure Streamlit
st.set_page_config(
    page_title="Health Report Analyzer",
    page_icon="🏥",
    layout="wide"
)

# Initialize Gemini API
def initialize_gemini():
    """Initialize Gemini API with API key"""
    api_key=st.secrets["GEMINI_API_KEY"]
    if not api_key:
        st.error("⚠️ Please set your GOOGLE_API_KEY environment variable")
        st.stop()
    
    genai.configure(api_key=api_key)
    
    # Try different model names in order of preference
    models_to_try = [
        'gemini-2.5-pro-preview-05-06',
        'gemini-1.5-pro'
    ]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            # Test the model with a simple prompt
            test_response = model.generate_content("Hello")
            # st.success(f"✅ Successfully connected to {model_name}")
            return model
        except Exception as e:
            continue
    
    st.stop()

class ReportAnalyzer:
    def __init__(self):
        self.model = initialize_gemini()
        self.report_text = ""
        self.report_chunks = []
        self.embeddings_model = None
        self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize embeddings model for RAG"""
        try:
            # Use Gemini's embedding model
            self.embeddings_model = genai.embed_content
        except Exception as e:
            st.warning(f"Embeddings model not available: {str(e)}")
    
    def _create_report_hash(self, text: str) -> str:
        """Create hash of report text to detect changes"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for RAG"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence end
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + len(chunk)
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= text_length:
                break
                
        return chunks
    
    def _get_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Get most relevant chunks for the query using simple keyword matching"""
        if not self.report_chunks:
            return []
        
        # Simple keyword-based relevance scoring
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for chunk in self.report_chunks:
            chunk_words = set(chunk.lower().split())
            # Calculate overlap score
            overlap = len(query_words.intersection(chunk_words))
            score = overlap / len(query_words) if query_words else 0
            chunk_scores.append((score, chunk))
        
        # Sort by relevance and return top chunks
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in chunk_scores[:top_k] if score > 0]
    
    def process_new_report(self, report_text: str) -> bool:
        """Process new report and check if it's different from current one"""
        new_hash = self._create_report_hash(report_text)
        current_hash = getattr(self, '_current_report_hash', None)
        
        if new_hash != current_hash:
            # New report detected
            self.report_text = report_text
            self.report_chunks = self._chunk_text(report_text)
            self._current_report_hash = new_hash
            return True  # Report changed
        
        return False  # Same report
        
    def extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def generate_summary(self, report_text: str) -> str:
        """Generate a comprehensive summary of the health report"""
        prompt = """
        You are a medical professional. Analyze this health report and provide a comprehensive summary.
        
        Include:
        1. Key test results and their values
        2. Normal vs abnormal findings
        3. Overall health status
        4. Important observations
        
        Format your response clearly with headers and bullet points.
        
        Report:
        {report_text}
        """.format(report_text=report_text)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def identify_areas_of_concern(self, report_text: str) -> str:
        """Identify and highlight areas that need medical attention"""
        prompt = """
        You are a medical professional analyzing a health report to identify areas of concern.
        
        Carefully review this health report and identify:
        
        1. **CRITICAL CONCERNS** (Immediate medical attention needed):
           - Values significantly outside normal ranges
           - Results indicating serious health conditions
           - Emergency situations
        
        2. **MODERATE CONCERNS** (Should be monitored/addressed):
           - Values slightly outside normal ranges
           - Trending patterns that could become problematic
           - Risk factors for future health issues
        
        3. **AREAS TO MONITOR** (Watch closely):
           - Values at the borderline of normal ranges
           - Results that could indicate developing conditions
           - Lifestyle factors affecting health
        
        For each concern, provide:
        - The specific test result and value
        - Normal reference range
        - What this could indicate
        - Recommended action (urgent care, follow-up, lifestyle changes)
        - Potential health implications if left untreated
        
        Use clear formatting with:
        - 🚨 for critical concerns
        - ⚠️ for moderate concerns  
        - 👀 for areas to monitor
        
        If no significant concerns are found, clearly state this and highlight positive findings.
        
        Report:
        {report_text}
        """.format(report_text=report_text)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error analyzing areas of concern: {str(e)}"
    
    def simplify_report(self, report_text: str) -> str:
        """Simplify the report for easy understanding"""
        prompt = """
        You are explaining medical results to a patient with no medical background.
        
        Take this health report and explain it in simple, easy-to-understand language:
        
        1. Avoid medical jargon
        2. Use everyday language
        3. Explain what each result means for their health
        4. Use analogies when helpful
        5. Organize by body systems or test types
        
        Report:
        {report_text}
        """.format(report_text=report_text)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error simplifying report: {str(e)}"
    
    def generate_diet_plan(self, report_text: str) -> str:
        """Generate personalized diet recommendations"""
        prompt = """
        You are a clinical nutritionist. Based on this health report, create a personalized diet plan.
        
        Analyze the report for:
        1. Abnormal values that can be improved with diet
        2. Nutritional deficiencies
        3. Risk factors (cholesterol, blood sugar, etc.)
        
        Provide:
        1. Specific foods to include and avoid
        2. 7-day meal plan suggestions
        3. Nutritional goals based on the findings
        4. Portion guidelines
        5. Supplement recommendations if needed
        
        Format with clear sections and actionable advice.
        
        Report:
        {report_text}
        """.format(report_text=report_text)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating diet plan: {str(e)}"
    
    def chat_response(self, question: str, report_context: str) -> str:
        """Generate response to user questions using RAG approach"""
        # Get relevant chunks from the report
        relevant_chunks = self._get_relevant_chunks(question, top_k=3)
        
        if relevant_chunks:
            # Use relevant chunks as context
            context = "\n\n".join([f"Relevant section {i+1}:\n{chunk}" 
                                 for i, chunk in enumerate(relevant_chunks)])
        else:
            # Fallback to first part of report if no relevant chunks found
            context = report_context[:2000] + "..." if len(report_context) > 2000 else report_context
        
        prompt = """
        You are a medical assistant helping a patient understand their health report.
        
        Use the following relevant sections from the patient's health report to answer their question:
        
        {context}
        
        Patient's Question: {question}
        
        Instructions:
        1. Answer based ONLY on the information provided in the relevant sections above
        2. If the question cannot be answered from the provided context, clearly state this
        3. Be specific and reference the actual values/findings from the report
        4. Keep your response conversational and supportive
        5. If you mention specific test results, include the values and reference ranges when available
        
        Response:
        """.format(context=context, question=question)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    st.title("🏥 Health Report Analyzer")
    st.markdown("Upload your health report and get AI-powered insights!")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ReportAnalyzer()
    if 'report_processed' not in st.session_state:
        st.session_state.report_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📄 Upload Report")
        uploaded_file = st.file_uploader(
            "Choose your health report",
            type=['pdf', 'txt'],
            help="Upload PDF or text file of your health report"
        )
        
        if uploaded_file and st.button("🔍 Analyze Report"):
            with st.spinner("Processing your report..."):
                # Extract text
                if uploaded_file.type == "application/pdf":
                    report_text = st.session_state.analyzer.extract_text_from_pdf(uploaded_file)
                else:
                    report_text = uploaded_file.getvalue().decode()
                
                if report_text:
                    # Check if this is a new report
                    is_new_report = st.session_state.analyzer.process_new_report(report_text)
                    
                    if is_new_report:
                        # Reset chat history for new report
                        st.session_state.chat_history = []
                        st.info("🔄 New report detected - Chat history has been reset")
                    
                    st.session_state.report_processed = True
                    st.success("✅ Report processed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Could not extract text from the file")
    
    # Main content area
    if not st.session_state.report_processed:
        # Welcome screen
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            ### 📋 Report Summary
            Get a comprehensive overview of all your test results
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Areas of Concern
            Identify issues that need medical attention
            """)
        
        with col3:
            st.markdown("""
            ### 🔍 Simplified Explanation
            Understand your results in plain English
            """)
        
        with col4:
            st.markdown("""
            ### 🥗 Diet Plan
            Get personalized nutrition recommendations
            """)
        
        st.markdown("---")
        st.info("👆 Upload your health report using the sidebar to get started!")
        
    else:
        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Summary", 
            "⚠️ Areas of Concern", 
            "🔍 Simplified", 
            "🥗 Diet Plan", 
            "💬 Chat"
        ])
        
        with tab1:
            st.header("📋 Report Summary")
            with st.spinner("Generating summary..."):
                summary = st.session_state.analyzer.generate_summary(
                    st.session_state.analyzer.report_text
                )
                st.markdown(summary)
        
        with tab2:
            st.header("⚠️ Areas of Concern")
            st.markdown("*Critical health indicators that need attention*")
            with st.spinner("Analyzing areas of concern..."):
                concerns = st.session_state.analyzer.identify_areas_of_concern(
                    st.session_state.analyzer.report_text
                )
                st.markdown(concerns)
            
            # Add disclaimer
            st.warning("""
            **⚠️ Medical Disclaimer:** This analysis is for informational purposes only and should not replace professional medical advice. 
            Always consult with your healthcare provider for proper medical interpretation and treatment decisions.
            """)
        
        with tab3:
            st.header("🔍 Simplified Explanation")
            with st.spinner("Simplifying your report..."):
                simplified = st.session_state.analyzer.simplify_report(
                    st.session_state.analyzer.report_text
                )
                st.markdown(simplified)
        
        with tab4:
            st.header("🥗 Personalized Diet Plan")
            with st.spinner("Creating your diet plan..."):
                diet_plan = st.session_state.analyzer.generate_diet_plan(
                    st.session_state.analyzer.report_text
                )
                st.markdown(diet_plan)
        
        with tab5:
            st.header("💬 Chat Assistant")
            st.markdown("Ask questions about your health report!")
            
            # Display chat history
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**You:** {question}")
                    st.markdown(f"**Assistant:** {answer}")
                    st.markdown("---")
            
            # Chat input
            with st.form("chat_form", clear_on_submit=True):
                user_question = st.text_input(
                    "Ask a question about your report:",
                    placeholder="e.g., What does my cholesterol level mean?"
                )
                submit_button = st.form_submit_button("Send")
                
                if submit_button and user_question:
                    with st.spinner("Thinking..."):
                        response = st.session_state.analyzer.chat_response(
                            user_question,
                            st.session_state.analyzer.report_text
                        )
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_question, response))
                        st.rerun()

if __name__ == "__main__":
    main()