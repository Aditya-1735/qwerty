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
    :root {
        --primary-color: #2E86AB;
        --background-color: #FFFFFF;
        --secondary-background-color: #F0F2F6;
        --text-color: #262730;
    }
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
    .research-card {
        background: linear-gradient(135deg, #00b894, #00a085);
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
    if 'research_queries' not in st.session_state:
        st.session_state.research_queries = []

# Perplexity AI Integration
class PerplexityAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
    
    def generate_response(self, prompt: str, model: str = "sonar-pro") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"I'm having trouble connecting right now. Status: {response.status_code}. Please try again later."
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Connection error: {str(e)}"
        except Exception as e:
            return f"I'm currently unavailable. Error: {str(e)}"

# Initialize Perplexity AI
try:
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    perplexity_ai = PerplexityAI(PERPLEXITY_API_KEY)
except KeyError:
    st.error("Perplexity API key not found in secrets. Please add PERPLEXITY_API_KEY to your Streamlit secrets.")
    st.stop()

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
    mood_types = [entry['mood_type'] for entry in st.session_state.mood_data[-7:]]
    avg_mood = sum(recent_moods) / len(recent_moods)
    
    prompt = f"""
    As a mental health companion, analyze this mood data and provide supportive insights:
    Recent mood scores (1-10): {recent_moods}
    Recent mood types: {mood_types}
    Average mood this week: {avg_mood:.1f}
    
    Provide a brief, encouraging analysis with actionable suggestions (2-3 sentences max).
    Focus on mental health best practices and evidence-based recommendations.
    Include any patterns you notice and suggest specific coping strategies.
    """
    
    return perplexity_ai.generate_response(prompt)

# Mindfulness Exercise Functions
def get_mindfulness_exercise(exercise_type: str, duration: int):
    prompt = f"""
    Create a {duration}-minute {exercise_type} mindfulness exercise.
    Provide step-by-step guidance that's calming and professional.
    Format as a guided meditation script with clear instructions.
    Base this on evidence-based mindfulness practices and techniques from current research.
    Include breathing cues and timing suggestions.
    Keep it concise but comprehensive.
    """
    
    return perplexity_ai.generate_response(prompt)

# CBT Tools Functions
def analyze_thought_pattern(negative_thought: str):
    prompt = f"""
    As a CBT-trained assistant, help analyze this thought pattern using evidence-based cognitive behavioral therapy techniques:
    Thought: "{negative_thought}"
    
    1. Identify potential cognitive distortions based on CBT literature (e.g., catastrophizing, all-or-nothing thinking)
    2. Suggest 2-3 balanced alternative thoughts using CBT reframing techniques
    3. Provide a brief coping strategy from established CBT practices
    
    Keep response supportive and professional (under 200 words). Base recommendations on current CBT research and best practices.
    Use a compassionate, non-judgmental tone.
    """
    
    return perplexity_ai.generate_response(prompt)

def get_coping_strategies(situation: str):
    prompt = f"""
    Provide 3 evidence-based coping strategies for this situation based on current mental health research:
    "{situation}"
    
    Focus on practical, actionable techniques from CBT, mindfulness, DBT, and other validated therapeutic approaches.
    Include recent research findings on effective coping mechanisms.
    Keep each strategy brief but specific with implementation steps.
    Prioritize techniques that can be used immediately.
    """
    
    return perplexity_ai.generate_response(prompt)

def get_research_insights(topic: str):
    prompt = f"""
    Summarize the latest research and evidence-based findings on {topic} in mental health from the past 2-3 years.
    Focus on practical applications and recent scientific developments.
    Keep it accessible for general audiences while maintaining scientific accuracy.
    Limit to 3-4 key insights with actionable takeaways.
    Include any breakthrough treatments or interventions if applicable.
    """
    
    return perplexity_ai.generate_response(prompt, model="sonar-pro")

# Main App Interface
def main():
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🧠 Mental Health Companion</div>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dashboard", "Mood Tracking", "Mindfulness", "CBT Tools", "Analytics", "Research Hub"]
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
    elif page == "Research Hub":
        research_hub()

def dashboard():
    st.header("Your Wellness Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Research Queries</h3>
            <h2>{}</h2>
            <p>Knowledge accessed</p>
        </div>
        """.format(len(st.session_state.research_queries)), unsafe_allow_html=True)
    
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
            <p><strong>Notes:</strong> {latest_mood['notes'][:100]}{'...' if len(latest_mood['notes']) > 100 else ''}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Start tracking your mood to see your progress here!")
    
    # AI Insights and Research
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AI Mood Insights")
        if st.button("Get Personalized Insights"):
            with st.spinner("Analyzing your mood patterns..."):
                insights = get_mood_insights()
                st.success(insights)
    
    with col2:
        st.subheader("Latest Research")
        research_topic = st.selectbox("Get insights on:", [
            "anxiety management techniques",
            "depression treatment advances",
            "mindfulness and neuroplasticity",
            "sleep and mental health connection",
            "stress reduction interventions",
            "digital mental health tools",
            "trauma-informed therapy",
            "exercise and mood regulation"
        ])
        
        if st.button("Get Research Insights"):
            with st.spinner("Fetching latest research..."):
                research = get_research_insights(research_topic)
                st.session_state.research_queries.append({
                    'date': datetime.now().isoformat(),
                    'topic': research_topic
                })
                st.info(research)

def mood_tracking():
    st.header("Mood Tracking & Journaling")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Track Your Mood")
        
        mood_score = st.slider("How are you feeling today?", 1, 10, 5, 
                              help="1 = Very Low, 10 = Excellent")
        
        mood_type = st.selectbox("Mood Category", [
            "Anxious", "Happy", "Sad", "Excited", "Stressed", 
            "Calm", "Angry", "Content", "Overwhelmed", "Grateful",
            "Frustrated", "Hopeful", "Lonely", "Energetic", "Peaceful"
        ])
        
        notes = st.text_area("What's on your mind?", 
                           placeholder="Describe your day, thoughts, or anything you'd like to remember...")
        
        if st.button("Save Mood Entry"):
            save_mood_entry(mood_score, mood_type, notes)
            st.success("Mood entry saved successfully! 🎉")
            st.balloons()
    
    with col2:
        st.subheader("Recent Entries")
        
        if st.session_state.mood_data:
            recent_entries = st.session_state.mood_data[-5:]
            for entry in reversed(recent_entries):
                mood_emoji = "😊" if entry['mood_score'] >= 7 else "😐" if entry['mood_score'] >= 4 else "😔"
                st.markdown(f"""
                <div class="mood-card">
                    <p><strong>{entry['date'][:10]}</strong> {mood_emoji} Score: {entry['mood_score']}/10</p>
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
            "Sleep Meditation",
            "Anxiety Relief",
            "Focus Enhancement",
            "Gratitude Practice"
        ])
        
        duration = st.select_slider("Duration (minutes)", 
                                  options=[3, 5, 10, 15, 20, 30], value=5)
        
        if st.button("Start Mindfulness Session"):
            with st.spinner("Preparing your personalized session..."):
                exercise = get_mindfulness_exercise(exercise_type, duration)
                st.session_state.current_exercise = exercise
                st.session_state.current_exercise_type = exercise_type
                st.session_state.current_exercise_duration = duration
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
                <h4>{st.session_state.current_exercise_type} - {st.session_state.current_exercise_duration} minutes</h4>
                <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                    {st.session_state.current_exercise.replace('\n', '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Complete Session ✅"):
                    st.success("Great job! Session completed. 🧘‍♀️")
                    del st.session_state.current_exercise
                    del st.session_state.current_exercise_type
                    del st.session_state.current_exercise_duration
                    st.rerun()
            
            with col_b:
                if st.button("End Early"):
                    st.info("Session ended. Take care! 💙")
                    del st.session_state.current_exercise
                    del st.session_state.current_exercise_type
                    del st.session_state.current_exercise_duration
                    st.rerun()
        else:
            st.subheader("Session History")
            if st.session_state.mindfulness_sessions:
                recent_sessions = st.session_state.mindfulness_sessions[-3:]
                for session in reversed(recent_sessions):
                    st.markdown(f"""
                    <div class="mood-card">
                        <p><strong>{session['date'][:10]}</strong></p>
                        <p>{session['type']} - {session['duration']} minutes</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No sessions completed yet. Start your first session above!")

def cbt_tools():
    st.header("Cognitive Behavioral Therapy Tools")
    
    tab1, tab2, tab3 = st.tabs(["Thought Analysis", "Coping Strategies", "Mood Patterns"])
    
    with tab1:
        st.subheader("Analyze Your Thoughts")
        st.write("Identify and challenge negative thought patterns with AI guidance based on CBT principles.")
        
        negative_thought = st.text_area("What negative thought are you experiencing?",
                                      placeholder="e.g., 'I always mess everything up' or 'Nobody likes me'",
                                      height=100)
        
        if st.button("Analyze Thought Pattern"):
            if negative_thought:
                with st.spinner("Analyzing thought pattern using CBT techniques..."):
                    analysis = analyze_thought_pattern(negative_thought)
                    st.session_state.cbt_sessions.append({
                        'date': datetime.now().isoformat(),
                        'type': 'thought_analysis',
                        'content': negative_thought
                    })
                    
                    st.markdown(f"""
                    <div class="cbt-card">
                        <h4>CBT Thought Analysis</h4>
                        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                            {analysis.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a thought to analyze.")
    
    with tab2:
        st.subheader("Coping Strategies")
        st.write("Get personalized coping strategies for challenging situations based on evidence-based practices.")
        
        situation = st.text_area("Describe the situation you're dealing with:",
                                placeholder="e.g., 'Work presentation anxiety' or 'Relationship conflict'",
                                height=100)
        
        if st.button("Get Coping Strategies"):
            if situation:
                with st.spinner("Generating personalized evidence-based strategies..."):
                    strategies = get_coping_strategies(situation)
                    st.session_state.cbt_sessions.append({
                        'date': datetime.now().isoformat(),
                        'type': 'coping_strategies',
                        'content': situation
                    })
                    
                    st.markdown(f"""
                    <div class="cbt-card">
                        <h4>Evidence-Based Coping Strategies</h4>
                        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                            {strategies.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please describe your situation.")
    
    with tab3:
        st.subheader("Mood Pattern Recognition")
        
        if st.session_state.mood_data:
            if st.button("Analyze My Mood Patterns"):
                with st.spinner("Analyzing your mood patterns for insights..."):
                    insights = get_mood_insights()
                    st.markdown(f"""
                    <div class="cbt-card">
                        <h4>Your Mood Pattern Analysis</h4>
                        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                            {insights.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Track your mood for a few days to see pattern analysis here.")

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
                    title="Mood Types Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekly patterns
    st.subheader("Weekly Patterns")
    df['weekday'] = pd.to_datetime(df['date']).dt.day_name()
    weekday_mood = df.groupby('weekday')['mood_score'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    fig = px.bar(x=weekday_mood.index, y=weekday_mood.values,
                title="Average Mood by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Average Mood Score'})
    fig.update_traces(marker_color='#74b9ff')
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
        st.metric("Tracking Streak", f"{streak} days")

def research_hub():
    st.header("Mental Health Research Hub")
    st.write("Stay updated with the latest evidence-based mental health research and insights powered by real-time AI.")
    
    tab1, tab2, tab3 = st.tabs(["Research Topics", "Custom Research", "Treatment Updates"])
    
    with tab1:
        st.subheader("Explore Research Areas")
        
        research_areas = [
            "Latest anxiety treatment approaches",
            "Depression and neuroscience findings",
            "Mindfulness and brain plasticity",
            "Digital mental health interventions", 
            "Trauma-informed therapy advances",
            "Sleep disorders and mental health",
            "Exercise psychology research",
            "Social media impact on mental health",
            "ADHD treatment innovations",
            "Bipolar disorder management",
            "PTSD therapy breakthroughs",
            "Eating disorder recovery methods"
        ]
        
        selected_topic = st.selectbox("Choose a research area:", research_areas)
        
        if st.button("Get Latest Research", key="research_topic"):
            with st.spinner("Searching latest research databases..."):
                research_summary = get_research_insights(selected_topic)
                st.session_state.research_queries.append({
                    'date': datetime.now().isoformat(),
                    'topic': selected_topic,
                    'type': 'predefined'
                })
                st.markdown(f"""
                <div class="research-card">
                    <h4>Research Summary: {selected_topic}</h4>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                        {research_summary.replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Custom Research Query")
        
        custom_query = st.text_area("Ask about specific mental health research:", 
                                   placeholder="e.g., 'effectiveness of CBT for social anxiety in teenagers'",
                                   height=100)
        
        if st.button("Search Research", key="custom_research"):
            if custom_query:
                with st.spinner("Searching research databases..."):
                    results = get_research_insights(custom_query)
                    st.session_state.research_queries.append({
                        'date': datetime.now().isoformat(),
                        'topic': custom_query,
                        'type': 'custom'
                    })
                    st.markdown(f"""
                    <div class="research-card">
                        <h4>Research Results</h4>
                        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                            {results.replace('\n', '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a research question.")
    
    with tab3:
        st.subheader("Treatment & Therapy Updates")
        
        treatment_topics = [
            "New FDA approved mental health medications",
            "Innovative therapy techniques 2024-2025",
            "Telehealth mental health effectiveness",
            "AI and machine learning in mental health",
            "Psychedelic therapy research updates",
            "Personalized medicine in psychiatry"
        ]
        
        treatment_topic = st.selectbox("Select treatment area:", treatment_topics)
        
        if st.button("Get Treatment Updates", key="treatment_updates"):
            with st.spinner("Fetching latest treatment information..."):
                updates = get_research_insights(treatment_topic)
                st.session_state.research_queries.append({
                    'date': datetime.now().isoformat(),
                    'topic': treatment_topic,
                    'type': 'treatment'
                })
                st.markdown(f"""
                <div class="research-card">
                    <h4>Treatment Updates: {treatment_topic}</h4>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                        {updates.replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Research History
    st.subheader("Your Research History")
    if st.session_state.research_queries:
        recent_queries = st.session_state.research_queries[-5:]
        for query in reversed(recent_queries):
            st.markdown(f"""
            <div class="mood-card">
                <p><strong>{query['date'][:10]}</strong> - {query['type'].title()}</p>
                <p><em>{query['topic']}</em></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Your research queries will appear here.")

# Footer
def footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>💙 Mental Health Companion - Your AI-powered wellness partner with real-time research insights</p>
        <p><small>If you're experiencing a mental health crisis, please contact emergency services or a mental health professional immediately.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()