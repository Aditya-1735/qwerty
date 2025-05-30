import streamlit as st
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import google.generativeai as genai
from dataclasses import dataclass
from fuzzywuzzy import fuzz

# Configure the page
st.set_page_config(
    page_title="AI Symptom Checker",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data structures
@dataclass
class Symptom:
    name: str
    severity: int  # 1-10 scale
    duration: str
    weight: float = 1.0  # Diagnostic weight
    additional_info: str = ""

@dataclass
class Condition:
    name: str
    confidence: float
    symptoms_match: List[str]
    advice: str
    urgency: str  # "low", "medium", "high", "emergency"
    precision: float = 0.0
    recall: float = 0.0

class SymptomChecker:
    def __init__(self):
        # Enhanced critical symptoms list
        self.critical_symptoms = [
            "chest pain", "shortness of breath", "difficulty breathing",
            "severe headache", "loss of consciousness", "confusion",
            "severe abdominal pain", "vomiting blood", "severe bleeding",
            "stroke symptoms", "severe allergic reaction", "anaphylaxis",
            "severe burns", "poisoning", "overdose"
        ]
        
        # Enhanced condition rules with symptom weights and alternatives
        self.condition_rules = {
            "Common Cold": {
                "symptoms": {
                    "runny nose": 2.0,
                    "sneezing": 1.5,
                    "headache": 1.0,
                    "sore throat": 1.5,
                    "nasal congestion": 2.0,
                    "congestion": 1.8,
                    "stuffy nose": 1.8,
                    "throat": 1.3,
                    "nose": 1.5
                },
                "confidence_threshold": 0.25,
                "advice": "Rest, stay hydrated, and use over-the-counter medications for symptom relief. Usually resolves in 7-10 days.",
                "urgency": "low"
            },
            "Flu": {
                "symptoms": {
                    "fever": 3.0,
                    "headache": 2.0,
                    "body aches": 3.0,
                    "fatigue": 2.5,
                    "chills": 2.0,
                    "muscle pain": 2.0,
                    "tired": 2.0,
                    "aches": 2.5,
                    "weakness": 2.0,
                    "exhausted": 2.2
                },
                "confidence_threshold": 0.3,
                "advice": "Rest, increase fluid intake, monitor temperature. Consider antiviral medication if within 48 hours of symptom onset.",
                "urgency": "medium"
            },
            "COVID-19": {
                "symptoms": {
                    "fever": 2.5,
                    "cough": 3.0,
                    "loss of taste": 4.0,
                    "loss of smell": 4.0,
                    "fatigue": 2.0,
                    "shortness of breath": 3.5,
                    "tired": 1.8,
                    "breathing problems": 3.0,
                    "smell": 3.5,
                    "taste": 3.5
                },
                "confidence_threshold": 0.25,
                "advice": "Self-isolate immediately, get tested for COVID-19, and monitor symptoms closely. Contact healthcare provider if symptoms worsen.",
                "urgency": "medium"
            },
            "Migraine": {
                "symptoms": {
                    "headache": 4.0,
                    "severe headache": 4.5,
                    "nausea": 2.0,
                    "sensitivity to light": 3.0,
                    "sensitivity to sound": 2.5,
                    "visual disturbances": 2.0,
                    "head pain": 3.5,
                    "light sensitivity": 2.8,
                    "sound sensitivity": 2.3
                },
                "confidence_threshold": 0.35,
                "advice": "Rest in a dark, quiet room. Apply cold compress. Consider prescribed migraine medication. Avoid triggers.",
                "urgency": "medium"
            },
            "Food Poisoning": {
                "symptoms": {
                    "nausea": 3.0,
                    "vomiting": 3.5,
                    "diarrhea": 3.5,
                    "abdominal pain": 3.0,
                    "fever": 2.0,
                    "stomach pain": 2.8,
                    "upset stomach": 2.5,
                    "stomach": 2.3,
                    "belly pain": 2.8
                },
                "confidence_threshold": 0.3,
                "advice": "Stay hydrated with clear fluids. Avoid solid foods initially. Seek medical attention if symptoms worsen or persist >48 hours.",
                "urgency": "medium"
            },
            "Emergency Condition": {
                "symptoms": {
                    "chest pain": 5.0,
                    "shortness of breath": 4.5,
                    "severe bleeding": 5.0,
                    "loss of consciousness": 5.0,
                    "stroke symptoms": 5.0,
                    "difficulty breathing": 4.0,
                    "breathing problems": 4.0,
                    "chest": 4.2,
                    "breathing": 3.8
                },
                "confidence_threshold": 0.1,
                "advice": "SEEK IMMEDIATE MEDICAL ATTENTION - Call emergency services now!",
                "urgency": "emergency"
            }
        }

    def check_critical_symptoms(self, symptoms: List[str]) -> bool:
        """Enhanced critical symptom detection with fuzzy matching"""
        if not symptoms:
            return False
            
        symptoms_text = " ".join(symptoms).lower()
        
        for critical in self.critical_symptoms:
            # Exact match
            if critical in symptoms_text:
                return True
            # Fuzzy match for typos/variations
            for symptom in symptoms:
                if fuzz.partial_ratio(critical, symptom.lower()) > 80:
                    return True
        return False

    def parse_symptoms_text(self, text: str) -> List[str]:
        """Enhanced text parsing for multiple separators"""
        if not text.strip():
            return []
        
        # Split by common separators
        separators = [',', ';', ' and ', ' & ', '\n', '.']
        symptoms = [text]
        
        for sep in separators:
            new_symptoms = []
            for symptom in symptoms:
                new_symptoms.extend([s.strip() for s in symptom.split(sep) if s.strip()])
            symptoms = new_symptoms
        
        # Clean up symptoms
        cleaned = []
        for symptom in symptoms:
            symptom = re.sub(r'^(i have|i am|experiencing|feeling)', '', symptom.lower()).strip()
            symptom = re.sub(r'[^\w\s]', '', symptom).strip()
            if symptom and len(symptom) > 2:
                cleaned.append(symptom.title())
        
        return list(set(cleaned))  # Remove duplicates

    def calculate_condition_confidence(self, user_symptoms: List[str], condition_symptoms: Dict[str, float]) -> Tuple[float, List[str], float, float]:
        """Enhanced confidence calculation with weighted symptoms and flexible matching"""
        if not user_symptoms or not condition_symptoms:
            return 0.0, [], 0.0, 0.0
        
        user_symptoms_lower = [s.lower().strip() for s in user_symptoms]
        matched_symptoms = []
        total_weight = sum(condition_symptoms.values())
        matched_weight = 0.0
        
        # Calculate weighted matches with more flexible matching
        for condition_symptom, weight in condition_symptoms.items():
            found_match = False
            condition_symptom_lower = condition_symptom.lower()
            best_match_score = 0
            
            for user_symptom in user_symptoms_lower:
                # More flexible matching logic
                match_score = 0
                
                # Exact match (highest score)
                if condition_symptom_lower == user_symptom:
                    match_score = 1.0
                # Substring match
                elif condition_symptom_lower in user_symptom or user_symptom in condition_symptom_lower:
                    match_score = 0.9
                # Fuzzy match for similar words
                elif fuzz.partial_ratio(condition_symptom_lower, user_symptom) > 70:
                    match_score = 0.8
                # Word-level matching (split and check individual words)
                else:
                    condition_words = set(condition_symptom_lower.split())
                    user_words = set(user_symptom.split())
                    word_overlap = len(condition_words.intersection(user_words))
                    if word_overlap > 0:
                        match_score = 0.7 * (word_overlap / max(len(condition_words), len(user_words)))
                
                if match_score > best_match_score:
                    best_match_score = match_score
            
            if best_match_score > 0.5:  # Lowered threshold for matches
                if condition_symptom not in matched_symptoms:
                    matched_symptoms.append(condition_symptom)
                    matched_weight += weight * best_match_score
                    found_match = True
        
        # Simplified confidence calculation
        if matched_weight > 0:
            confidence = matched_weight / total_weight
            precision = confidence
            recall = len(matched_symptoms) / len(user_symptoms) if user_symptoms else 0
        else:
            confidence = 0.0
            precision = 0.0
            recall = 0.0
        
        return confidence, matched_symptoms, precision, recall

    def predict_conditions(self, symptoms: List[str]) -> List[Condition]:
        """Enhanced condition prediction with better scoring"""
        if not symptoms:
            return []
            
        conditions = []
        
        for condition_name, condition_data in self.condition_rules.items():
            confidence, matched_symptoms, precision, recall = self.calculate_condition_confidence(
                symptoms, condition_data["symptoms"]
            )
            
            # Dynamic threshold adjustment based on urgency
            threshold = condition_data["confidence_threshold"]
            
            if condition_data["urgency"] == "emergency":
                threshold *= 0.7
            elif condition_data["urgency"] == "high":
                threshold *= 0.8
            
            if confidence >= threshold:
                conditions.append(Condition(
                    name=condition_name,
                    confidence=confidence,
                    symptoms_match=matched_symptoms,
                    advice=condition_data["advice"],
                    urgency=condition_data["urgency"],
                    precision=precision,
                    recall=recall
                ))
        
        # Sort by confidence, then by urgency priority
        urgency_priority = {"emergency": 4, "high": 3, "medium": 2, "low": 1}
        return sorted(conditions, 
                     key=lambda x: (x.confidence, urgency_priority.get(x.urgency, 0)), 
                     reverse=True)

    def generate_followup_questions(self, symptoms: List[str], api_key: str = None) -> List[str]:
        """Enhanced follow-up question generation"""
        if not api_key:
            # Enhanced mock questions based on symptoms
            base_questions = [
                "How long have you been experiencing these symptoms?",
                "On a scale of 1-10, how severe would you rate your symptoms?",
                "Have you noticed any triggers that make the symptoms worse?",
                "Are you currently taking any medications or have any allergies?"
            ]
            
            symptoms_text = " ".join(symptoms).lower()
            specific_questions = []
            
            if any(word in symptoms_text for word in ["fever", "temperature"]):
                specific_questions.append("What is your current temperature and how long have you had a fever?")
            
            if any(word in symptoms_text for word in ["headache", "head pain"]):
                specific_questions.append("Where exactly is the headache located and is it throbbing or constant?")
            
            if any(word in symptoms_text for word in ["cough", "coughing"]):
                specific_questions.append("Is your cough dry or producing mucus? What color is the mucus if any?")
            
            if any(word in symptoms_text for word in ["pain", "ache", "hurt"]):
                specific_questions.append("Does the pain radiate to other areas? What makes it better or worse?")
            
            if any(word in symptoms_text for word in ["nausea", "vomiting", "stomach"]):
                specific_questions.append("When did the nausea/vomiting start and have you eaten anything unusual recently?")
            
            if any(word in symptoms_text for word in ["breathing", "breath", "chest"]):
                specific_questions.append("Are you having difficulty breathing at rest or only with activity?")
            
            return base_questions[:2] + specific_questions[:3]
        
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-pro')
                
                prompt = f"""
                You are an experienced medical assistant helping to gather comprehensive symptom information. 
                
                Patient's reported symptoms: {', '.join(symptoms)}
                
                Generate 4-5 targeted medical follow-up questions that would help:
                1. Determine symptom duration, severity, and progression
                2. Identify potential triggers or risk factors
                3. Differentiate between similar conditions
                4. Assess urgency and severity
                5. Gather relevant medical history
                
                Guidelines:
                - Ask about timing (when started, how long, getting worse/better)
                - Ask about severity and impact on daily activities
                - Ask about associated symptoms not yet mentioned
                - Ask about recent travel, exposure, or changes
                - Make questions conversational and patient-friendly
                - Avoid medical jargon
                - Don't repeat information about symptoms already reported
                
                Return exactly 4-5 questions, one per line, without numbering or bullet points.
                Each question should end with a question mark.
                """
                
                response = model.generate_content(prompt)
                questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
                
                cleaned_questions = []
                for q in questions:
                    q = re.sub(r'^[\d\.\-\*\•\s]+', '', q).strip()
                    if q and q.endswith('?') and len(q) > 10:
                        cleaned_questions.append(q)
                
                return cleaned_questions[:5] if cleaned_questions else self.generate_followup_questions(symptoms)
            
            except Exception as e:
                st.error(f"Error generating questions with Gemini: {e}")
                return self.generate_followup_questions(symptoms)

    def interpret_followup_answers(self, questions_answers: Dict[str, str], api_key: str = None) -> Dict[str, str]:
        """Enhanced answer interpretation with structured extraction"""
        if not api_key or not questions_answers:
            mock_interpretation = {}
            
            for question, answer in questions_answers.items():
                answer_lower = answer.lower()
                
                if "how long" in question.lower() or "when" in question.lower():
                    if any(word in answer_lower for word in ["day", "days"]):
                        numbers = re.findall(r'\d+', answer)
                        if numbers:
                            mock_interpretation["Duration"] = f"{numbers[0]} days"
                    elif any(word in answer_lower for word in ["hour", "hours"]):
                        numbers = re.findall(r'\d+', answer)
                        if numbers:
                            mock_interpretation["Duration"] = f"{numbers[0]} hours"
                    elif any(word in answer_lower for word in ["week", "weeks"]):
                        numbers = re.findall(r'\d+', answer)
                        if numbers:
                            mock_interpretation["Duration"] = f"{numbers[0]} weeks"
                
                if "scale" in question.lower() or "severe" in question.lower():
                    numbers = re.findall(r'\b([1-9]|10)\b', answer)
                    if numbers:
                        mock_interpretation["Severity"] = f"{numbers[0]}/10"
                
                if "temperature" in question.lower() or "fever" in question.lower():
                    temp_match = re.search(r'(\d{2,3}(?:\.\d)?)', answer)
                    if temp_match:
                        mock_interpretation["Temperature"] = f"{temp_match.group(1)}°F"
            
            return mock_interpretation
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            context = ""
            for question, answer in questions_answers.items():
                context += f"Q: {question}\nA: {answer}\n\n"
            
            prompt = f"""
            Analyze the following patient responses and extract key medical information.
            
            Patient Responses:
            {context}
            
            Extract and return ONLY the following information that is explicitly mentioned:
            
            1. Duration: Convert any time references to standardized format (e.g., "2 days", "6 hours", "1 week")
            2. Severity: Any numerical ratings mentioned (e.g., "7/10", "moderate", "severe")
            3. Temperature: Exact temperature readings if provided (include units)
            4. Triggers: Specific triggers, activities, or situations that worsen symptoms
            5. Associated symptoms: Any NEW symptoms mentioned in answers (not in original list)
            6. Medications: Current medications, recent medication changes, or allergies mentioned
            7. Recent activities: Travel, exposure to sick people, dietary changes, stress
            8. Progression: Whether symptoms are getting better, worse, or staying the same
            9. Impact: How symptoms affect daily activities, work, or sleep
            
            Format your response as:
            Key: Value
            
            Only include information that was explicitly stated. If nothing is mentioned for a category, skip it.
            Be specific and concise. Use the patient's exact words when possible.
            """
            
            response = model.generate_content(prompt)
            
            interpretation = {}
            lines = response.text.strip().split('\n')
            
            for line in lines:
                if ':' in line and len(line.strip()) > 3:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        if value and value != "None" and value != "N/A":
                            interpretation[key] = value
            
            return interpretation
            
        except Exception as e:
            st.error(f"Error interpreting answers with Gemini: {e}")
            return self.interpret_followup_answers(questions_answers)

def initialize_session_state():
    """Initialize session state variables with proper cleanup"""
    if 'step' not in st.session_state:
        st.session_state.step = 'initial'
    if 'symptoms' not in st.session_state:
        st.session_state.symptoms = []
    if 'followup_questions' not in st.session_state:
        st.session_state.followup_questions = []
    if 'followup_answers' not in st.session_state:
        st.session_state.followup_answers = {}
    if 'interpretation' not in st.session_state:
        st.session_state.interpretation = {}
    if 'checker' not in st.session_state:
        st.session_state.checker = SymptomChecker()
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None

def reset_session():
    """Complete session reset"""
    keys_to_reset = ['step', 'symptoms', 'followup_questions', 'followup_answers', 'interpretation', 'emergency_acknowledged']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.checker = SymptomChecker()

def main():
    initialize_session_state()
    
    st.title("🏥 AI-Powered Symptom Checker")
    st.markdown("*Get personalized health insights through intelligent symptom analysis*")
    
    with st.expander("⚠️ Important Medical Disclaimer", expanded=False):
        st.warning("""
        **This tool is for informational purposes only and should not replace professional medical advice.**
        
        - Always consult with healthcare professionals for proper diagnosis
        - In case of emergency symptoms, call emergency services immediately
        - This tool uses AI predictions which may not be 100% accurate
        - Results should be used as guidance only, not as medical diagnosis
        - This is a demonstration tool for hackathon purposes
        """)
    
    with st.sidebar:
        st.header("🔧 Settings")
        
        try:
            api_key = st.secrets.get("GEMINI_API_KEY") or None
            if api_key:
                st.success("🤖 AI-Enhanced Analysis Active")
                # st.info("Powered by Gemini 1.5 Pro")
            else:
                raise KeyError("No API key found")
        except:
            api_key = None
            st.warning("⚠️ AI features unavailable")
            st.info("Using enhanced mock responses")
        
        st.header("📊 Session Info")
        st.info(f"Current Step: {st.session_state.step.title()}")
        st.info(f"Symptoms Entered: {len(st.session_state.symptoms)}")
        
        if st.button("🔄 Reset Session"):
            reset_session()
            st.rerun()
    
    st.session_state.api_key = api_key
    
    if st.session_state.step == 'initial':
        initial_symptom_input()
    elif st.session_state.step == 'followup':
        followup_questions_interface()
    elif st.session_state.step == 'results':
        display_results()

def initial_symptom_input():
    """Enhanced initial symptom input interface"""
    st.header("📝 Tell us about your symptoms")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symptom_text = st.text_area(
            "Describe your symptoms:",
            placeholder="e.g., I have a severe headache and fever for 2 days, also feeling nauseous...",
            height=120,
            help="You can use commas, 'and', or separate lines to list multiple symptoms"
        )
        
        st.subheader("🔘 Or select from common symptoms:")
        
        symptom_categories = {
            "General": ["Fever", "Fatigue", "Chills", "Night Sweats"],
            "Head & Throat": ["Headache", "Sore Throat", "Runny Nose", "Loss of Taste", "Loss of Smell"],
            "Respiratory": ["Cough", "Shortness of Breath", "Chest Pain", "Wheezing"],
            "Digestive": ["Nausea", "Vomiting", "Diarrhea", "Abdominal Pain", "Loss of Appetite"],
            "Musculoskeletal": ["Body Aches", "Joint Pain", "Back Pain", "Muscle Pain"],
            "Other": ["Dizziness", "Skin Rash", "Swelling", "Confusion"]
        }
        
        selected_symptoms = []
        
        for category, symptoms in symptom_categories.items():
            with st.expander(f"{category} Symptoms"):
                cols = st.columns(2)
                for i, symptom in enumerate(symptoms):
                    with cols[i % 2]:
                        if st.checkbox(symptom, key=f"symptom_{category}_{i}"):
                            selected_symptoms.append(symptom)
    
    with col2:
        st.subheader("🚨 Emergency Symptoms")
        st.error("""
        **Seek immediate medical attention if you have:**
        - Severe chest pain or pressure
        - Difficulty breathing or shortness of breath
        - Loss of consciousness or fainting
        - Severe bleeding that won't stop
        - Signs of stroke (facial drooping, arm weakness, speech difficulty)
        - Severe allergic reaction (swelling, difficulty breathing)
        - High fever with severe headache and neck stiffness
        - Severe abdominal pain
        """)
        
        st.info("""
        **When in doubt, it's better to seek medical care.**
        
        Emergency: 911
        Non-emergency: Contact your healthcare provider
        """)
    
    all_symptoms = []
    
    if symptom_text.strip():
        parsed_symptoms = st.session_state.checker.parse_symptoms_text(symptom_text)
        all_symptoms.extend(parsed_symptoms)
    
    all_symptoms.extend(selected_symptoms)
    
    seen = set()
    unique_symptoms = []
    for symptom in all_symptoms:
        if symptom.lower() not in seen:
            seen.add(symptom.lower())
            unique_symptoms.append(symptom)
    
    if unique_symptoms:
        st.subheader("📋 Your Current Symptoms:")
        symptom_display = " | ".join([f"**{s}**" for s in unique_symptoms])
        st.markdown(symptom_display)
    
    if st.button("🔍 Analyze Symptoms", 
                type="primary", 
                disabled=len(unique_symptoms) == 0,
                help="Click to start AI-powered symptom analysis"):
        
        if unique_symptoms:
            st.session_state.symptoms = unique_symptoms
            
            if 'emergency_acknowledged' not in st.session_state:
                st.session_state.emergency_acknowledged = False
            
            if st.session_state.checker.check_critical_symptoms(unique_symptoms):
                st.error("🚨 **EMERGENCY ALERT**: You have reported symptoms that may require immediate medical attention!")
                st.error("**Please contact emergency services (911) or visit the nearest emergency room immediately!**")
                
                if not st.session_state.emergency_acknowledged:
                    if st.button("⚠️ I understand - Continue anyway", type="secondary"):
                        st.session_state.emergency_acknowledged = True
                        st.warning("Proceeding with analysis, but please seek immediate medical care.")
                        st.rerun()
                    else:
                        st.stop()
            
            if not st.session_state.checker.check_critical_symptoms(unique_symptoms) or st.session_state.get('emergency_acknowledged', False):
                with st.spinner("🤖 Generating AI-powered follow-up questions..."):
                    time.sleep(1.5)
                    questions = st.session_state.checker.generate_followup_questions(
                        unique_symptoms, 
                        api_key=st.session_state.get('api_key')
                    )
                    st.session_state.followup_questions = questions
                    st.session_state.step = 'followup'
                    if 'emergency_acknowledged' in st.session_state:
                        del st.session_state.emergency_acknowledged
                    st.rerun()

def followup_questions_interface():
    """Enhanced follow-up questions interface"""
    st.header("📋 Additional Information Needed")
    st.markdown("*Please answer these questions to help provide more accurate recommendations:*")
    
    with st.expander("🎯 Your reported symptoms:", expanded=True):
        cols = st.columns(min(len(st.session_state.symptoms), 4))
        for i, symptom in enumerate(st.session_state.symptoms):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div style='background-color:#e3f2fd;padding:8px 12px;margin:4px;border-radius:15px;text-align:center;border-left:4px solid #2196f3;'>
                    <strong>{symptom}</strong>
                </div>
                """, unsafe_allow_html=True)
    
    st.progress(0.5, text="Step 2 of 3: Gathering additional information")
    
    st.markdown("### 📝 Please answer the following questions:")
    answers = {}
    
    for i, question in enumerate(st.session_state.followup_questions):
        with st.container():
            st.markdown(f"**Question {i+1} of {len(st.session_state.followup_questions)}**")
            answer = st.text_area(
                question,
                key=f"followup_{i}",
                placeholder="Please provide as much detail as possible. Be specific about timing, severity, and any patterns you've noticed...",
                height=80
            )
            if answer.strip():
                answers[question] = answer
            st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("← Back to Symptoms", help="Go back to modify your symptoms"):
            st.session_state.step = 'initial'
            st.rerun()
    
    with col2:
        if st.button("Skip Questions →", help="Continue without additional information"):
            st.session_state.followup_answers = {}
            st.session_state.step = 'results'
            st.rerun()
    
    with col3:
        answers_provided = len(answers)
        button_text = f"Get Results → ({answers_provided} answers provided)"
        
        if st.button(button_text, type="primary", help="Proceed to analysis results"):
            st.session_state.followup_answers = answers
            
            if st.session_state.get('api_key') and answers:
                with st.spinner("🤖 Analyzing your responses with advanced AI..."):
                    time.sleep(2)
                    interpretation = st.session_state.checker.interpret_followup_answers(
                        answers, 
                        st.session_state.get('api_key')
                    )
                    st.session_state.interpretation = interpretation
            
            st.session_state.step = 'results'
            st.rerun()

def display_results():
    """Enhanced results display"""
    st.header("🎯 Your Comprehensive Symptom Analysis")
    
    st.progress(1.0, text="Step 3 of 3: Analysis complete")
    
    with st.expander("📊 Complete Symptom Profile", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Primary Symptoms:")
            for i, symptom in enumerate(st.session_state.symptoms, 1):
                st.markdown(f"{i}. **{symptom}**")
        
        with col2:
            st.subheader("📝 Additional Information:")
            if st.session_state.followup_answers:
                if st.session_state.get('interpretation'):
                    st.markdown("**🤖 AI-Extracted Key Information:**")
                    for key, value in st.session_state.interpretation.items():
                        st.markdown(f"• **{key}**: {value}")
                    st.markdown("---")
                
                st.markdown("**📋 Your Responses Summary:**")
                for i, (question, answer) in enumerate(st.session_state.followup_answers.items(), 1):
                    st.markdown(f"**Q{i}:** {question}")
                    st.markdown(f"**A{i}:** {answer[:100]}{'...' if len(answer) > 100 else ''}")
                    st.markdown("---")
            else:
                st.info("No additional information provided")
    
    if st.session_state.followup_answers:
        st.subheader("📝 Detailed Question & Answer Review")
        for i, (question, answer) in enumerate(st.session_state.followup_answers.items(), 1):
            with st.expander(f"Question {i}: {question[:60]}{'...' if len(question) > 60 else ''}"):
                st.write(f"**Question:** {question}")
                st.write(f"**Full Answer:** {answer}")
    
    with st.spinner("🔬 Performing comprehensive symptom analysis..."):
        time.sleep(2)
        conditions = st.session_state.checker.predict_conditions(st.session_state.symptoms)
    
    if conditions:
        st.subheader("🔍 Possible Conditions Analysis")
        st.markdown("*Results are ranked by confidence and clinical relevance*")
        
        for i, condition in enumerate(conditions[:4]):
            urgency_colors = {
                "low": ("🟢", "#4caf50"),
                "medium": ("🟡", "#ff9800"), 
                "high": ("🟠", "#ff5722"),
                "emergency": ("🔴", "#f44336")
            }
            
            icon, color = urgency_colors.get(condition.urgency, ("⚪", "#9e9e9e"))
            
            with st.container():
                st.markdown(f"""
                <div style='background: linear-gradient(90deg, {color}20, transparent); 
                           border-left: 5px solid {color}; 
                           padding: 15px; 
                           border-radius: 10px; 
                           margin: 10px 0;'>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"### {i+1}. {condition.name}")
                    if condition.symptoms_match:
                        st.markdown(f"**Matched symptoms:** {', '.join(condition.symptoms_match)}")
                    else:
                        st.markdown("**Symptoms considered in analysis**")
                
                with col2:
                    st.metric("Confidence", f"{condition.confidence:.0%}")
                
                with col3:
                    st.metric("Precision", f"{condition.precision:.0%}")
                
                with col4:
                    st.markdown(f"**{icon} {condition.urgency.title()}**")
                
                if condition.urgency == "emergency":
                    st.error(f"🚨 **URGENT ACTION REQUIRED:** {condition.advice}")
                elif condition.urgency == "high":
                    st.warning(f"⚠️ **Important:** {condition.advice}")
                elif condition.urgency == "medium":
                    st.info(f"💡 **Recommendation:** {condition.advice}")
                else:
                    st.success(f"✅ **Guidance:** {condition.advice}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")
    
    else:
        st.warning("🤔 **Inconclusive Analysis**")
        st.info("""
        Based on the provided symptoms, we cannot determine specific conditions with sufficient confidence. 
        This could mean:
        - Symptoms are very mild or general
        - More information is needed for accurate assessment
        - Professional medical evaluation is recommended
        
        **Recommendation:** Consult with a healthcare professional for proper evaluation.
        """)
    
    st.subheader("📋 General Health Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚨 Seek Immediate Care If:
        - Symptoms worsen rapidly or suddenly
        - High fever (>103°F/39.4°C) that won't break
        - Severe difficulty breathing
        - Chest pain or pressure
        - Severe or persistent pain
        - Signs of dehydration (dizziness, dry mouth, little/no urination)
        - Any symptom that concerns you significantly
        """)
    
    with col2:
        st.markdown("""
        ### 🏠 Self-Care Recommendations:
        - **Hydration**: Drink plenty of fluids (water, herbal tea, clear broths)
        - **Rest**: Get adequate sleep and avoid strenuous activities
        - **Monitoring**: Keep track of symptoms and temperature
        - **Nutrition**: Eat light, easy-to-digest foods
        - **Isolation**: Avoid contact with others if potentially infectious
        - **Documentation**: Keep a symptom diary for healthcare visits
        """)
    
    st.markdown("---")
    st.subheader("🎯 What would you like to do next?")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 New Assessment", help="Start completely over with new symptoms"):
            reset_session()
            st.rerun()
    
    with col2:
        if st.button("📝 Modify Symptoms", help="Go back and change your symptoms"):
            st.session_state.step = 'initial'
            st.rerun()
    
    with col3:
        if st.button("📞 Find Healthcare", help="Get information about finding medical care"):
            st.session_state.show_healthcare_resources = not st.session_state.get('show_healthcare_resources', False)
    
    with col4:
        if st.button("💾 Save Report", help="Download or save your symptom analysis"):
            report = f"""
            SYMPTOM ANALYSIS REPORT
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            SYMPTOMS REPORTED:
            {chr(10).join([f"• {s}" for s in st.session_state.symptoms])}
            
            ANALYSIS RESULTS:
            {chr(10).join([f"• {c.name}: {c.confidence:.0%} confidence ({c.urgency} urgency)" for c in conditions[:3]])}
            
            RECOMMENDATIONS:
            Follow the guidance provided in the analysis above.
            
            DISCLAIMER:
            This is an AI-generated analysis for informational purposes only.
            Always consult healthcare professionals for medical advice.
            """
            
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"symptom_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    if st.session_state.get('show_healthcare_resources'):
        st.subheader("🏥 Healthcare Resources")
        st.info("""
        **For this hackathon demo, this would integrate with:**
        - Healthcare provider location APIs
        - Insurance network directories  
        - Telemedicine platforms
        - Local clinic and hospital databases
        - Appointment scheduling systems
        
        **Immediate Resources:**
        - Emergency: 911
        - Poison Control: 1-800-222-1222
        - Mental Health Crisis: 988
        """)

if __name__ == "__main__":
    main()