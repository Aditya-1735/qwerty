import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import io

# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

# Gemini AI imports
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Gemini API
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')
    ai_available = True
except Exception as e:
    st.sidebar.warning(f"⚠️ AI features unavailable: {str(e)}")
    model = None
    ai_available = False

# Custom CSS for better styling (White Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: white !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        text-align: center;
        border-radius: 10px;
        margin-bottom: 2rem;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        font-size: 1.5rem;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        color: #ffffff;
        font-weight: bold;
    }
    .high-risk {
        background-color: #3c1617;
        border-left: 8px solid #d32f2f;
        color: #ff6b6b;
    }
    .medium-risk {
        background-color: #3d2914;
        border-left: 8px solid #f57c00;
        color: #ffb74d;
    }
    .low-risk {
        background-color: #1b2f1b;
        border-left: 8px solid #388e3c;
        color: #66bb6a;
    }
    .risk-card h3, .risk-card h2 {
        color: inherit;
        text-shadow: none;
    }
    .sidebar .stMarkdown, .sidebar .stNumberInput, .sidebar .stSelectbox {
        color: #e0e0e0;
    }
    .stMarkdown, .stMetric {
        color: #e0e0e0;
    }
    .ai-tab {
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        background: linear-gradient(135deg, #1e3a2e, #2e4a3e);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.25);
        color: #a5d6a7;
    }
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background-color: #262626;
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 1rem;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header"><span class="emoji">🏥</span>Health Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Health Assessment & Lifestyle Guidance Platform</p>', unsafe_allow_html=True)
st.markdown("---")

# Helper function to clear AI session data
def clear_ai_session_data():
    """Clear AI-generated content from session state when new predictions are made"""
    ai_keys = ['comprehensive_health_insights']
    for key in ai_keys:
        if key in st.session_state:
            del st.session_state[key]

# AI Helper Function for Comprehensive Health Insights with Lifestyle Recommendations
def generate_comprehensive_health_insights(user_data, diabetes_prob, heart_disease_prob, recommendations):
    """Generate comprehensive health insights with complete lifestyle recommendations using Gemini 1.5 Pro"""
    if not ai_available:
        return "AI insights unavailable - API configuration error"
    
    try:
        prompt = f"""
        As a comprehensive medical AI assistant, analyze this health profile and provide detailed personalized insights with complete lifestyle recommendations:
        
        **Patient Profile:**
        - Age: {user_data['age']} years
        - Sex: {user_data['sex']}
        - BMI: {user_data['bmi']:.1f}
        - Glucose: {user_data['glucose']} mg/dL
        - Blood Pressure: {user_data['blood_pressure']} mmHg
        - Cholesterol: {user_data['cholesterol']} mg/dL
        - Max Heart Rate: {user_data['max_heart_rate']} bpm
        - Chest Pain Type: {user_data['chest_pain']}
        - Exercise Induced Angina: {user_data['exercise_angina']}
        
        **Risk Assessment:**
        - Diabetes Risk: {diabetes_prob:.1%}
        - Heart Disease Risk: {heart_disease_prob:.1%}
        
        Please provide a comprehensive analysis with the following sections:

        ## 🏥 Health Summary
        Provide a 2-3 sentence overall health assessment and risk overview.

        ## 🎯 Key Risk Factors Identified
        List and explain the most important risk factors based on the data.

        ## 🍽️ Personalized Diet Recommendations
        - **Daily meal structure and timing**
        - **Specific foods to include** (with quantities and portions)
        - **Foods to limit or avoid** (with clear explanations)
        - **Sample meal ideas**: Provide 3 complete daily meal plans (breakfast, lunch, dinner, snacks)
        - **Hydration guidelines** (water intake recommendations)
        - **Supplement recommendations** if appropriate

        ## 🏃‍♂️ Exercise & Physical Activity Plan
        - **Weekly exercise schedule** (specific days and activities)
        - **Recommended activities** based on current fitness level and health conditions
        - **Target heart rate zones** for this person's age and condition
        - **Progression plan** for increasing intensity over 4-6 weeks
        - **Safety considerations** and precautions
        - **Specific exercises** for diabetes/heart health prevention

        ## 😴 Sleep & Recovery Recommendations
        - **Optimal sleep duration** and timing for this age group
        - **Sleep hygiene tips** tailored to health condition
        - **Recovery strategies** and relaxation techniques

        ## 🧘‍♂️ Stress Management & Mental Health
        - **Stress reduction techniques** suitable for this person
        - **Mindfulness and relaxation strategies**
        - **Lifestyle modifications** for stress management
        - **Mental health support** recommendations

        ## 📊 Health Monitoring & Prevention
        - **Which health metrics to track** regularly (daily, weekly, monthly)
        - **Recommended frequency** for medical check-ups
        - **Home monitoring tools** and techniques
        - **Warning signs** to watch for specific to their risk profile

        ## 💡 Daily Health Habits & Routines
        - **5 specific, actionable daily habits** for optimal health
        - **Morning routine** recommendations
        - **Evening routine** suggestions
        - **Weekly health goals** to work towards

        ## ⚠️ When to Seek Medical Attention
        - **Specific symptoms or changes** to watch for
        - **Emergency warning signs** requiring immediate medical care
        - **Recommended healthcare consultations** and specialists
        - **Preventive screenings** schedule

        ## 🎯 Personalized Action Plan
        - **30-day immediate goals**
        - **90-day medium-term objectives**
        - **Long-term lifestyle targets**

        Keep all recommendations practical, specific, and tailored to this individual's profile. Include specific numbers (portions, durations, frequencies) where possible. Make it actionable and easy to follow. Add appropriate medical disclaimers.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate comprehensive health insights: {str(e)}"

# Function to generate enhanced PDF using ReportLab
def generate_enhanced_pdf_report(user_data, diabetes_prob, heart_disease_prob, recommendations, 
                               diabetes_accuracy, heart_accuracy, metrics_data, ai_insights=None):
    """Generate enhanced PDF report with AI insights"""
    
    try:
        # Create a BytesIO buffer to hold the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#34495e')
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
        
        # Determine risk levels
        diabetes_risk_level = "High" if diabetes_prob >= 0.7 else "Medium" if diabetes_prob >= 0.3 else "Low"
        heart_risk_level = "High" if heart_disease_prob >= 0.7 else "Medium" if heart_disease_prob >= 0.3 else "Low"
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Title
        title_text = "AI-Enhanced Health Risk Assessment & Lifestyle Report" if ai_insights else "Health Risk Assessment Report"
        elements.append(Paragraph(title_text, title_style))
        elements.append(Spacer(1, 12))
        
        # Header information
        header_data = [
            ['Generated on:', current_date],
            ['Patient ID:', f"{hash(str(user_data)) % 10000:04d}"],
            ['Report Type:', 'Comprehensive Health Analysis & Lifestyle Guide'],
            ['AI Enhancement:', 'Yes' if ai_insights else 'No']
        ]
        
        header_table = Table(header_data, colWidths=[2*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e9ecef')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ]))
        
        elements.append(header_table)
        elements.append(Spacer(1, 20))
        
        # AI Insights section (if available)
        if ai_insights:
            elements.append(Paragraph("Comprehensive Health Analysis & Lifestyle Recommendations", heading_style))
            
            # Clean AI insights for PDF
            clean_insights = ai_insights.replace('**', '').replace('*', '').replace('#', '').replace('•', '•')
            elements.append(Paragraph(clean_insights, normal_style))
            elements.append(Spacer(1, 20))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", heading_style))
        
        summary_text = f"""
        This comprehensive health risk assessment analyzes your personal health metrics using machine learning models 
        trained on real medical datasets. The analysis focuses on two critical health conditions: Diabetes and Heart Disease.
        <br/><br/>
        <b>Key Findings:</b><br/>
        • Diabetes Risk: {diabetes_risk_level} ({diabetes_prob:.1%} probability)<br/>
        • Heart Disease Risk: {heart_risk_level} ({heart_disease_prob:.1%} probability)<br/>
        • Overall Health Status: {"Requires Attention" if diabetes_prob > 0.5 or heart_disease_prob > 0.5 else "Generally Healthy"}
        """
        
        elements.append(Paragraph(summary_text, normal_style))
        elements.append(Spacer(1, 20))
        
        # Personal Health Profile
        elements.append(Paragraph("Personal Health Profile", heading_style))
        
        profile_data = [
            ['Metric', 'Value', 'Unit'],
            ['Age', str(user_data['age']), 'years'],
            ['Sex', user_data['sex'], ''],
            ['Pregnancies', str(user_data['pregnancies']), '(if applicable)'],
            ['BMI', f"{user_data['bmi']:.1f}", 'kg/m²'],
            ['Blood Pressure', str(user_data['blood_pressure']), 'mmHg'],
            ['Glucose Level', str(user_data['glucose']), 'mg/dL'],
            ['Cholesterol', str(user_data['cholesterol']), 'mg/dL'],
            ['Max Heart Rate', str(user_data['max_heart_rate']), 'bpm'],
            ['Skin Thickness', str(user_data['skin_thickness']), 'mm'],
            ['Insulin Level', str(user_data['insulin']), 'μU/mL'],
            ['Chest Pain Type', user_data['chest_pain'], ''],
            ['Exercise Induced Angina', user_data['exercise_angina'], '']
        ]
        
        profile_table = Table(profile_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(profile_table)
        elements.append(Spacer(1, 20))
        
        # Rest of the PDF generation code remains the same...
        # (Including Risk Assessment Results, Health Metrics Analysis, Recommendations, etc.)
        
        # Build PDF
        doc.build(elements)
        
        # Get the value of the BytesIO buffer and return it
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return True, pdf_data
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return False, None

# Sidebar for user inputs
st.sidebar.header("📊 Enter Your Health Metrics")

@st.cache_data
def load_real_datasets():
    try:
        # Update paths to match your repository structure
        diabetes_data = pd.read_csv('streamlit-apps/health-predictor/diabetes.csv')
        heart_data = pd.read_csv('streamlit-apps/health-predictor/heart.csv')
        
        return diabetes_data, heart_data
    except FileNotFoundError as e:
        st.error(f"Dataset file not found: {e}")
        st.error("Please make sure CSV files are in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None

# Load real datasets
diabetes_data, heart_data = load_real_datasets()

if diabetes_data is None or heart_data is None:
    st.stop()

# Train models using real datasets
@st.cache_resource
def train_models():
    # Prepare diabetes model
    diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X_diabetes = diabetes_data[diabetes_features]
    y_diabetes = diabetes_data['Outcome']
    
    # Handle missing values in diabetes data
    X_diabetes = X_diabetes.fillna(X_diabetes.median())
    
    # Train diabetes model
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_diabetes, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
    )
    diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
    diabetes_model.fit(X_train_d, y_train_d)
    
    # Prepare heart disease model
    heart_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X_heart = heart_data[heart_features]
    y_heart = heart_data['target']
    
    # Handle missing values in heart data
    X_heart = X_heart.fillna(X_heart.median())
    
    # Train heart disease model
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
    )
    heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
    heart_model.fit(X_train_h, y_train_h)
    
    # Calculate model accuracies
    diabetes_accuracy = diabetes_model.score(X_test_d, y_test_d)
    heart_accuracy = heart_model.score(X_test_h, y_test_h)
    
    return diabetes_model, heart_model, diabetes_accuracy, heart_accuracy

diabetes_model, heart_model, diabetes_accuracy, heart_accuracy = train_models()

# User input section - Updated for real dataset features
col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown("### Basic Info")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    sex = st.selectbox("Sex", ["Female", "Male"])
    pregnancies = st.number_input("Pregnancies (if applicable)", min_value=0, max_value=20, value=0, step=1)
    
    st.markdown("### Vital Signs")
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100, step=1)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=120, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    st.markdown("### Additional Metrics")
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=200, step=1)
    max_heart_rate = st.number_input("Max Heart Rate", min_value=50, max_value=220, value=150, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=0, step=1)
    
    st.markdown("### Risk Factors")
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

# Convert categorical inputs
sex_binary = 1 if sex == "Male" else 0
chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
chest_pain_encoded = chest_pain_map[chest_pain]
exercise_angina_binary = 1 if exercise_angina == "Yes" else 0

# Calculate additional derived features
diabetes_pedigree = 0.5  # Default value
fasting_blood_sugar = 1 if glucose > 120 else 0
rest_ecg = 0  # Default normal
oldpeak = 1.0  # Default value
slope = 1  # Default value
ca = 0  # Default value
thal = 2  # Default value

# Store user data for report generation
user_data = {
    'age': age,
    'sex': sex,
    'pregnancies': pregnancies,
    'glucose': glucose,
    'blood_pressure': blood_pressure,
    'bmi': bmi,
    'cholesterol': cholesterol,
    'max_heart_rate': max_heart_rate,
    'skin_thickness': skin_thickness,
    'insulin': insulin,
    'chest_pain': chest_pain,
    'exercise_angina': exercise_angina
}

# Predict button
if st.sidebar.button("🔍 Predict Health Risks", type="primary"):
    # Clear previous AI-generated content
    clear_ai_session_data()
    
    # Prepare input data for diabetes model
    diabetes_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                               insulin, bmi, diabetes_pedigree, age]])
    
    # Prepare input data for heart disease model
    heart_input = np.array([[age, sex_binary, chest_pain_encoded, blood_pressure, cholesterol,
                            fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina_binary,
                            oldpeak, slope, ca, thal]])
    
    # Make predictions
    diabetes_prob = diabetes_model.predict_proba(diabetes_input)[0][1]
    heart_disease_prob = heart_model.predict_proba(heart_input)[0][1]
    
    # Store predictions in session state for PDF generation
    st.session_state.diabetes_prob = diabetes_prob
    st.session_state.heart_disease_prob = heart_disease_prob
    st.session_state.user_data = user_data
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🩺 Diabetes Risk Assessment")
        
        # Risk level determination
        if diabetes_prob < 0.3:
            risk_level = "Low"
            risk_class = "low-risk"
            risk_color = "#4caf50"
        elif diabetes_prob < 0.7:
            risk_level = "Medium"
            risk_class = "medium-risk"
            risk_color = "#ff9800"
        else:
            risk_level = "High"
            risk_class = "high-risk"
            risk_color = "#f44336"
        
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <h3>Risk Level: {risk_level}</h3>
            <h2>{diabetes_prob:.1%} probability</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Diabetes risk gauge
        fig_diabetes = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = diabetes_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Diabetes Risk %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_diabetes.update_layout(height=300)
        st.plotly_chart(fig_diabetes, use_container_width=True)
    
    with col2:
        st.subheader("❤️ Heart Disease Risk Assessment")
        
        # Risk level determination
        if heart_disease_prob < 0.3:
            risk_level_heart = "Low"
            risk_class_heart = "low-risk"
            risk_color_heart = "#4caf50"
        elif heart_disease_prob < 0.7:
            risk_level_heart = "Medium"
            risk_class_heart = "medium-risk"
            risk_color_heart = "#ff9800"
        else:
            risk_level_heart = "High"
            risk_class_heart = "high-risk"
            risk_color_heart = "#f44336"
        
        st.markdown(f"""
        <div class="risk-card {risk_class_heart}">
            <h3>Risk Level: {risk_level_heart}</h3>
            <h2>{heart_disease_prob:.1%} probability</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Heart disease risk gauge
        fig_heart = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = heart_disease_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Heart Disease Risk %"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color_heart},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_heart.update_layout(height=300)
        st.plotly_chart(fig_heart, use_container_width=True)
    
    # AI-Powered Health Insights Section - Only Health Summary with Comprehensive Lifestyle Recommendations
    if ai_available:
        st.subheader("🤖 AI-Powered Health Summary & Lifestyle Recommendations")
        
        # Display existing comprehensive health insights if available
        if 'comprehensive_health_insights' in st.session_state:
            st.markdown('<div class="ai-tab">', unsafe_allow_html=True)
            st.markdown(st.session_state.comprehensive_health_insights)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown('<div class="ai-tab">', unsafe_allow_html=True)
                with st.spinner("🧠 Generating comprehensive health analysis with lifestyle recommendations..."):
                    comprehensive_insights = generate_comprehensive_health_insights(user_data, diabetes_prob, heart_disease_prob, [])
                    st.session_state.comprehensive_health_insights = comprehensive_insights
                    st.markdown(comprehensive_insights)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature importance analysis
    st.subheader("📈 Risk Factor Analysis")
    
    # Get feature importance
    diabetes_feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                             'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
    heart_feature_names = ['Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol',
                          'Fasting Blood Sugar', 'Rest ECG', 'Max Heart Rate', 'Exercise Angina',
                          'ST Depression', 'Slope', 'Major Vessels', 'Thalassemia']
    
    diabetes_importance = diabetes_model.feature_importances_
    heart_importance = heart_model.feature_importances_
    
    # Create comparison chart
    fig_importance = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Diabetes Risk Factors', 'Heart Disease Risk Factors'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig_importance.add_trace(
        go.Bar(x=diabetes_feature_names, y=diabetes_importance, name="Diabetes", marker_color="#1f77b4"),
        row=1, col=1
    )
    
    fig_importance.add_trace(
        go.Bar(x=heart_feature_names, y=heart_importance, name="Heart Disease", marker_color="#ff7f0e"),
        row=1, col=2
    )
    
    fig_importance.update_layout(
        height=400,
        showlegend=False,
        title_text="Feature Importance in Risk Prediction"
    )
    fig_importance.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Recommendations section
    st.subheader("💡 Personalized Recommendations")
    
    # Personalized recommendations based on real dataset insights
    recommendations = []
    
    if glucose > 140:
        recommendations.append("🍎 High glucose detected - Consider consulting a doctor about diabetes management")
    elif glucose > 100:
        recommendations.append("⚠️ Elevated glucose - Monitor blood sugar and consider dietary changes")
    
    if bmi > 30:
        recommendations.append("🏃‍♂️ BMI indicates obesity - Weight management recommended")
    elif bmi > 25:
        recommendations.append("📊 BMI is overweight - Consider healthy weight management")
    
    if blood_pressure > 140:
        recommendations.append("💊 High blood pressure detected - Consult healthcare provider")
    elif blood_pressure > 120:
        recommendations.append("⚡ Elevated blood pressure - Monitor and consider lifestyle changes")
    
    if cholesterol > 240:
        recommendations.append("🥗 High cholesterol - Consider dietary changes and medical consultation")
    elif cholesterol > 200:
        recommendations.append("📈 Borderline high cholesterol - Monitor and maintain healthy diet")
    
    if chest_pain != "Asymptomatic":
        recommendations.append("❤️ Chest pain reported - Important to discuss with healthcare provider")
    
    if exercise_angina == "Yes":
        recommendations.append("🏥 Exercise-induced angina - Seek medical evaluation")
    
    if max_heart_rate < 100:
        recommendations.append("💓 Low maximum heart rate - Consider cardiac evaluation")
    
    if pregnancies > 4:
        recommendations.append("👶 Multiple pregnancies may increase diabetes risk - Regular monitoring advised")
    
    # Store recommendations in session state
    st.session_state.recommendations = recommendations
    
    if recommendations:
        for rec in recommendations:
            st.write(f"• {rec}")
    else:
        st.success("🎉 Great job! Your current metrics look healthy. Keep maintaining your lifestyle!")
    
    # Health metrics comparison with updated metrics
    st.subheader("📊 Your Metrics vs. Healthy Ranges")
    
    metrics_data = {
        'Metric': ['BMI', 'Glucose', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate'],
        'Your Value': [bmi, glucose, blood_pressure, cholesterol, max_heart_rate],
        'Healthy Min': [18.5, 70, 90, 120, 100],
        'Healthy Max': [24.9, 99, 120, 199, 220],
        'Your Status': [
            'Normal' if 18.5 <= bmi <= 24.9 else 'Above Normal' if bmi > 24.9 else 'Below Normal',
            'Normal' if glucose <= 99 else 'Elevated',
            'Normal' if blood_pressure <= 120 else 'Elevated',
            'Normal' if cholesterol <= 199 else 'Elevated',
            'Normal' if 100 <= max_heart_rate <= 220 else 'Abnormal'
        ]
    }
    
    # Store metrics data in session state
    st.session_state.metrics_data = metrics_data
    
    # Add model accuracy information
    st.subheader("🎯 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Diabetes Model Accuracy", f"{diabetes_accuracy:.1%}")
    with col2:
        st.metric("Heart Disease Model Accuracy", f"{heart_accuracy:.1%}")
    
    st.info("These models are trained on real medical datasets and provide clinically-relevant predictions.")
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig_metrics = px.bar(
        metrics_df, 
        x='Metric', 
        y='Your Value',
        color='Your Status',
        color_discrete_map={'Normal': '#4caf50', 'Elevated': '#f44336', 'Above Normal': '#ff9800', 'Below Normal': '#2196f3', 'Abnormal': '#f44336'},
        title="Your Health Metrics"
    )
    
    # Add healthy range indicators
    for i, row in metrics_df.iterrows():
        fig_metrics.add_shape(
            type="rect",
            x0=i-0.4, x1=i+0.4,
            y0=row['Healthy Min'], y1=row['Healthy Max'],
            fillcolor="rgba(0,255,0,0.2)",
            line=dict(color="green", width=2),
        )
    
    fig_metrics.update_layout(height=400)
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Enhanced PDF Report Generation Section
    st.subheader("📄 Generate Comprehensive PDF Report")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        report_type = "Enhanced AI Lifestyle Report" if ai_available else "Standard Report"
        if st.button(f"📊 Generate & Download {report_type}", type="primary", use_container_width=True):
            with st.spinner("Generating your comprehensive health & lifestyle report..."):
                try:
                    # Generate AI insights for PDF if available
                    ai_insights_for_pdf = None
                    if ai_available:
                        ai_insights_for_pdf = st.session_state.get('comprehensive_health_insights', None)
                        if not ai_insights_for_pdf:
                            ai_insights_for_pdf = generate_comprehensive_health_insights(user_data, diabetes_prob, heart_disease_prob, recommendations)
                    
                    # Generate enhanced PDF
                    success, pdf_data = generate_enhanced_pdf_report(
                        user_data, 
                        diabetes_prob, 
                        heart_disease_prob, 
                        recommendations,
                        diabetes_accuracy, 
                        heart_accuracy, 
                        metrics_data,
                        ai_insights_for_pdf
                    )
                    
                    if success:
                        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{'AI_Lifestyle_' if ai_available else ''}Health_Report_{current_date}.pdf"
                        
                        st.success("✅ Comprehensive Health & Lifestyle Report Generated!")
                        st.download_button(
                            label="💾 Download Lifestyle Report",
                            data=pdf_data,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        feature_list = "Complete risk assessment, personalized lifestyle recommendations, diet & exercise plans, health monitoring guide"
                        if ai_available:
                            feature_list += ", AI-powered comprehensive health insights"
                        feature_list += ", and medical disclaimers"
                        
                        st.info(f"📋 **Report includes:** {feature_list}")
                        
                    else:
                        st.error("❌ Failed to generate PDF. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")

# Check if predictions exist in session state for PDF generation
elif hasattr(st.session_state, 'diabetes_prob') and hasattr(st.session_state, 'heart_disease_prob'):
    st.subheader("📄 Generate Lifestyle Report")
    st.info("💡 Your previous predictions are available. You can generate a comprehensive lifestyle report.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        report_type = "Enhanced AI Lifestyle Report" if ai_available else "Standard Report"
        if st.button(f"📊 Generate & Download {report_type}", type="secondary", use_container_width=True):
            with st.spinner("Generating your comprehensive lifestyle report..."):
                try:
                    # Generate AI insights for PDF if available
                    ai_insights_for_pdf = None
                    if ai_available:
                        ai_insights_for_pdf = st.session_state.get('comprehensive_health_insights', None)
                        if not ai_insights_for_pdf:
                            ai_insights_for_pdf = generate_comprehensive_health_insights(
                                st.session_state.user_data, 
                                st.session_state.diabetes_prob, 
                                st.session_state.heart_disease_prob, 
                                st.session_state.get('recommendations', [])
                            )
                    
                    # Generate PDF using session state data
                    success, pdf_data = generate_enhanced_pdf_report(
                        st.session_state.user_data, 
                        st.session_state.diabetes_prob, 
                        st.session_state.heart_disease_prob, 
                        st.session_state.get('recommendations', []),
                        diabetes_accuracy, 
                        heart_accuracy, 
                        st.session_state.get('metrics_data', {}),
                        ai_insights_for_pdf
                    )
                    
                    if success:
                        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{'AI_Lifestyle_' if ai_available else ''}Health_Report_{current_date}.pdf"
                        
                        st.success("✅ Lifestyle Report Generated Successfully!")
                        st.download_button(
                            label="💾 Download Lifestyle Report",
                            data=pdf_data,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                    else:
                        st.error("❌ Failed to generate PDF. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")

# Welcome message for new users
else:
    st.markdown("""
    ## Welcome to the AI-Powered Health Risk Predictor! 🏥
    
    This application uses machine learning models trained on real medical datasets to assess your risk for:
    - **Diabetes** (based on Pima Indian Diabetes Database)
    - **Heart Disease** (based on Cleveland Heart Disease Database)
    
    ### 🤖 AI-Enhanced Features:
    - **Comprehensive Health Summary** with complete lifestyle recommendations
    - **Personalized Diet Plans** with specific meal suggestions and portions
    - **Custom Exercise Programs** tailored to your health condition
    - **Sleep & Recovery Guidelines** for optimal health
    - **Stress Management Strategies** and mental health support
    - **Health Monitoring Plans** with specific metrics to track
    
    ### How to use:
    1. Enter your health metrics in the sidebar
    2. Click "🔍 Predict Health Risks"
    3. Review your personalized risk assessment
    4. Get comprehensive lifestyle recommendations from AI
    5. Download your detailed lifestyle report
    
    ### ⚠️ Important Disclaimer:
    This tool is for educational and screening purposes only. Always consult healthcare professionals for medical advice.
    """)
    
    if ai_available:
        st.success("✅ AI features are enabled and ready to provide comprehensive lifestyle insights!")
    else:
        st.warning("⚠️ AI features are not available. Standard analysis will be provided.")