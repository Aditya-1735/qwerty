import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import json
import time
import requests
from typing import Dict, List, Any
import re

# Configure page
st.set_page_config(
    page_title="Mental Health Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .mood-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .exercise-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .cbt-card {
        background: linear-gradient(135deg, #fd79a8, #e84393);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'mood_data' not in st.session_state:
        st.session_state.mood_data = []
    if 'journal_entries' not in st.session_state:
        st.session_state.journal_entries = []
    if 'mindfulness_sessions' not in st.session_state:
        st.session_state.mindfulness_sessions = []
    if 'cbt_sessions' not in st.session_state:
        st.session_state.cbt_sessions = []

# Gemini AI Integration
class GeminiAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-05-06:generateContent"
    
    def generate_response(self, prompt: str) -> str:
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "I'm having trouble connecting right now. Please try again later."
        except Exception as e:
            return "I'm currently unavailable. Please try again in a moment."


GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
gemini_ai = GeminiAI(GEMINI_API_KEY)

# Mood Tracking Functions
def save_mood_entry(mood_score: int, mood_type: str, notes: str):
    entry = {
        'date': datetime.now().isoformat(),
        'mood_score': mood_score,
        'mood_type': mood_type,
        'notes': notes
    }
    st.session_state.mood_data.append(entry)

def get_mood_insights():
    if not st.session_state.mood_data:
        return "No mood data available yet. Start tracking your mood to see insights!"
    
    recent_moods = [entry['mood_score'] for entry in st.session_state.mood_data[-7:]]
    avg_mood = sum(recent_moods) / len(recent_moods)
    
    prompt = f"""
    As a mental health companion, analyze this mood data and provide supportive insights:
    Recent mood scores (1-10): {recent_moods}
    Average mood this week: {avg_mood:.1f}
    
    Provide a brief, encouraging analysis with actionable suggestions (2-3 sentences max).
    """
    
    return gemini_ai.generate_response(prompt)

# Mindfulness Exercise Functions
def get_mindfulness_exercise(exercise_type: str, duration: int):
    prompt = f"""
    Create a {duration}-minute {exercise_type} mindfulness exercise.
    Provide step-by-step guidance that's calming and professional.
    Format as a guided meditation script with clear instructions.
    Keep it concise but comprehensive.
    """
    
    return gemini_ai.generate_response(prompt)

# CBT Tools Functions
def analyze_thought_pattern(negative_thought: str):
    prompt = f"""
    As a CBT-trained assistant, help analyze this thought pattern:
    Thought: "{negative_thought}"
    
    1. Identify potential cognitive distortions
    2. Suggest 2-3 balanced alternative thoughts
    3. Provide a brief coping strategy
    
    Keep response supportive and professional (under 200 words).
    """
    
    return gemini_ai.generate_response(prompt)

def get_coping_strategies(situation: str):
    prompt = f"""
    Provide 3 evidence-based coping strategies for this situation:
    "{situation}"
    
    Focus on practical, actionable techniques from CBT and mindfulness.
    Keep each strategy brief but specific.
    """
    
    return gemini_ai.generate_response(prompt)

# Main App Interface
def main():
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🧠 Mental Health Companion</div>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dashboard", "Mood Tracking", "Mindfulness", "CBT Tools", "Analytics"]
    )
    
    if page == "Dashboard":
        dashboard()
    elif page == "Mood Tracking":
        mood_tracking()
    elif page == "Mindfulness":
        mindfulness_exercises()
    elif page == "CBT Tools":
        cbt_tools()
    elif page == "Analytics":
        analytics()

def dashboard():
    st.header("Your Wellness Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Mood Entries</h3>
            <h2>{}</h2>
            <p>Total tracked</p>
        </div>
        """.format(len(st.session_state.mood_data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Mindfulness</h3>
            <h2>{}</h2>
            <p>Sessions completed</p>
        </div>
        """.format(len(st.session_state.mindfulness_sessions)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>CBT Sessions</h3>
            <h2>{}</h2>
            <p>Thoughts analyzed</p>
        </div>
        """.format(len(st.session_state.cbt_sessions)), unsafe_allow_html=True)
    
    # Recent Activity
    st.subheader("Recent Activity")
    
    if st.session_state.mood_data:
        latest_mood = st.session_state.mood_data[-1]
        st.markdown(f"""
        <div class="mood-card">
            <h4>Latest Mood Entry</h4>
            <p><strong>Score:</strong> {latest_mood['mood_score']}/10</p>
            <p><strong>Type:</strong> {latest_mood['mood_type']}</p>
            <p><strong>Date:</strong> {latest_mood['date'][:10]}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Start tracking your mood to see your progress here!")
    
    # AI Insights
    st.subheader("AI Insights")
    if st.button("Get Mood Insights"):
        with st.spinner("Analyzing your mood patterns..."):
            insights = get_mood_insights()
            st.success(insights)

def mood_tracking():
    st.header("Mood Tracking & Journaling")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Track Your Mood")
        
        mood_score = st.slider("How are you feeling today?", 1, 10, 5, 
                              help="1 = Very Low, 10 = Excellent")
        
        mood_type = st.selectbox("Mood Category", [
            "Anxious", "Happy", "Sad", "Excited", "Stressed", 
            "Calm", "Angry", "Content", "Overwhelmed", "Grateful"
        ])
        
        notes = st.text_area("What's on your mind?", 
                           placeholder="Describe your day, thoughts, or anything you'd like to remember...")
        
        if st.button("Save Mood Entry"):
            save_mood_entry(mood_score, mood_type, notes)
            st.success("Mood entry saved successfully!")
    
    with col2:
        st.subheader("Recent Entries")
        
        if st.session_state.mood_data:
            recent_entries = st.session_state.mood_data[-5:]
            for entry in reversed(recent_entries):
                st.markdown(f"""
                <div class="mood-card">
                    <p><strong>{entry['date'][:10]}</strong> - Score: {entry['mood_score']}/10</p>
                    <p><em>{entry['mood_type']}</em></p>
                    <p>{entry['notes'][:100]}{'...' if len(entry['notes']) > 100 else ''}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No mood entries yet. Start tracking above!")

def mindfulness_exercises():
    st.header("AI-Guided Mindfulness Exercises")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Choose Your Practice")
        
        exercise_type = st.selectbox("Exercise Type", [
            "Breathing Meditation",
            "Body Scan",
            "Loving-Kindness",
            "Mindful Walking",
            "Stress Relief",
            "Sleep Meditation"
        ])
        
        duration = st.select_slider("Duration (minutes)", 
                                  options=[3, 5, 10, 15, 20], value=5)
        
        if st.button("Start Mindfulness Session"):
            with st.spinner("Preparing your personalized session..."):
                exercise = get_mindfulness_exercise(exercise_type, duration)
                st.session_state.current_exercise = exercise
                st.session_state.mindfulness_sessions.append({
                    'date': datetime.now().isoformat(),
                    'type': exercise_type,
                    'duration': duration
                })
                st.rerun()
    
    with col2:
        if 'current_exercise' in st.session_state:
            st.subheader("Your Guided Session")
            st.markdown(f"""
            <div class="exercise-card">
                <h4>{exercise_type} - {duration} minutes</h4>
                <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                    {st.session_state.current_exercise.replace('\n', '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Complete Session"):
                st.success("Great job! Session completed. 🧘‍♀️")
                del st.session_state.current_exercise

def cbt_tools():
    st.header("Cognitive Behavioral Therapy Tools")
    
    tab1, tab2 = st.tabs(["Thought Analysis", "Coping Strategies"])
    
    with tab1:
        st.subheader("Analyze Your Thoughts")
        st.write("Identify and challenge negative thought patterns with AI guidance.")
        
        negative_thought = st.text_area("What negative thought are you experiencing?",
                                      placeholder="e.g., 'I always mess everything up' or 'Nobody likes me'")
        
        if st.button("Analyze Thought Pattern"):
            if negative_thought:
                with st.spinner("Analyzing thought pattern..."):
                    analysis = analyze_thought_pattern(negative_thought)
                    st.session_state.cbt_sessions.append({
                        'date': datetime.now().isoformat(),
                        'type': 'thought_analysis',
                        'content': negative_thought
                    })
                    
                    st.markdown(f"""
                    <div class="cbt-card">
                        <h4>Thought Analysis</h4>
                        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                            {analysis.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a thought to analyze.")
    
    with tab2:
        st.subheader("Coping Strategies")
        st.write("Get personalized coping strategies for challenging situations.")
        
        situation = st.text_area("Describe the situation you're dealing with:",
                                placeholder="e.g., 'Work presentation anxiety' or 'Relationship conflict'")
        
        if st.button("Get Coping Strategies"):
            if situation:
                with st.spinner("Generating personalized strategies..."):
                    strategies = get_coping_strategies(situation)
                    st.session_state.cbt_sessions.append({
                        'date': datetime.now().isoformat(),
                        'type': 'coping_strategies',
                        'content': situation
                    })
                    
                    st.markdown(f"""
                    <div class="cbt-card">
                        <h4>Coping Strategies</h4>
                        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                            {strategies.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please describe your situation.")

def analytics():
    st.header("Your Wellness Analytics")
    
    if not st.session_state.mood_data:
        st.info("Track your mood for at least a few days to see analytics.")
        return
    
    # Mood Trends
    df = pd.DataFrame(st.session_state.mood_data)
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mood Trend")
        fig = px.line(df, x='date', y='mood_score', 
                     title="Mood Score Over Time",
                     labels={'mood_score': 'Mood Score', 'date': 'Date'})
        fig.update_traces(line_color='#667eea', line_width=3)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Mood Distribution")
        mood_counts = df['mood_type'].value_counts()
        fig = px.pie(values=mood_counts.values, names=mood_counts.index,
                    title="Mood Types Distribution")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_mood = df['mood_score'].mean()
        st.metric("Average Mood", f"{avg_mood:.1f}/10")
    
    with col2:
        best_day = df.loc[df['mood_score'].idxmax(), 'date']
        st.metric("Best Day", str(best_day))
    
    with col3:
        total_entries = len(df)
        st.metric("Total Entries", total_entries)
    
    with col4:
        streak = len(df)  # Simplified streak calculation
        st.metric("Current Streak", f"{streak} days")

# Footer
def footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>💙 Mental Health Companion - Your AI-powered wellness partner</p>
        <p><small>Remember: This tool supports but doesn't replace professional mental health care.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()