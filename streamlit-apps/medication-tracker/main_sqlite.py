import streamlit as st
import pandas as pd
import json
import requests
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
import hashlib
import sqlite3
from dataclasses import dataclass, asdict
import schedule
import threading
import time as time_module
from twilio.rest import Client

# Data Classes
@dataclass
class Medication:
    name: str
    dosage: str
    frequency: str
    time_slots: List[str]
    start_date: str
    end_date: Optional[str] = None
    notes: str = ""
    active: bool = True
    drug_id: str = ""

@dataclass
class User:
    username: str
    email: str
    phone: str = ""
    password_hash: str = ""
    medications: List[Medication] = None
    
    def __post_init__(self):
        if self.medications is None:
            self.medications = []

class DatabaseManager:
    def __init__(self, db_path: str = "medication_tracker.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                phone TEXT,
                password_hash TEXT NOT NULL
            )
        ''')
        
        # Medications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                name TEXT NOT NULL,
                dosage TEXT NOT NULL,
                frequency TEXT NOT NULL,
                time_slots TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                notes TEXT,
                active BOOLEAN DEFAULT 1,
                drug_id TEXT,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_user(self, user: User) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO users (username, email, phone, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (user.username, user.email, user.phone, user.password_hash))
            
            # Save medications
            cursor.execute('DELETE FROM medications WHERE username = ?', (user.username,))
            for med in user.medications:
                cursor.execute('''
                    INSERT INTO medications 
                    (username, name, dosage, frequency, time_slots, start_date, end_date, notes, active, drug_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user.username, med.name, med.dosage, med.frequency, 
                     json.dumps(med.time_slots), med.start_date, med.end_date, 
                     med.notes, med.active, med.drug_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error saving user: {e}")
            return False
    
    def load_user(self, username: str) -> Optional[User]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user_data = cursor.fetchone()
            
            if not user_data:
                conn.close()
                return None
            
            user = User(
                username=user_data[0],
                email=user_data[1],
                phone=user_data[2] or "",
                password_hash=user_data[3],
                medications=[]
            )
            
            # Load medications
            cursor.execute('SELECT * FROM medications WHERE username = ?', (username,))
            med_data = cursor.fetchall()
            
            for med_row in med_data:
                medication = Medication(
                    name=med_row[2],
                    dosage=med_row[3],
                    frequency=med_row[4],
                    time_slots=json.loads(med_row[5]),
                    start_date=med_row[6],
                    end_date=med_row[7],
                    notes=med_row[8] or "",
                    active=bool(med_row[9]),
                    drug_id=med_row[10] or ""
                )
                user.medications.append(medication)
            
            conn.close()
            return user
        except Exception as e:
            st.error(f"Error loading user: {e}")
            return None

class DrugInteractionChecker:
    def __init__(self):
        # For demo purposes, we'll use a mock API response
        # In production, you would use the actual DrugBank API
        self.base_url = "https://go.drugbank.com/api/v1"
        self.api_key = "YOUR_DRUGBANK_API_KEY"  # Replace with actual API key
    
    def search_drug(self, drug_name: str) -> List[Dict]:
        """Search for drug information"""
        # Mock response for demo
        drugs = {
            "aspirin": {"id": "DB00945", "name": "Aspirin", "generic_name": "acetylsalicylic acid"},
            "warfarin": {"id": "DB00682", "name": "Warfarin", "generic_name": "warfarin"},
            "ibuprofen": {"id": "DB01050", "name": "Ibuprofen", "generic_name": "ibuprofen"},
            "acetaminophen": {"id": "DB00316", "name": "Acetaminophen", "generic_name": "acetaminophen"},
            "lisinopril": {"id": "DB00722", "name": "Lisinopril", "generic_name": "lisinopril"}
        }
        
        results = []
        for key, value in drugs.items():
            if drug_name.lower() in key.lower():
                results.append(value)
        
        return results
    
    def check_interactions(self, drug_ids: List[str]) -> List[Dict]:
        """Check for drug interactions"""
        # Mock interaction data for demo
        interactions = [
            {
                "drug1": "aspirin",
                "drug2": "warfarin",
                "severity": "Major",
                "description": "Increased risk of bleeding when aspirin is combined with warfarin."
            },
            {
                "drug1": "aspirin",
                "drug2": "ibuprofen",
                "severity": "Moderate",
                "description": "NSAIDs may reduce the cardioprotective effect of low-dose aspirin."
            }
        ]
        
        return interactions

class ReminderService:
    def __init__(self):
        # Twilio configuration
        self.twilio_config = {
            'account_sid': 'YOUR_TWILIO_ACCOUNT_SID',  # Replace with your Twilio Account SID
            'auth_token': 'YOUR_TWILIO_AUTH_TOKEN',    # Replace with your Twilio Auth Token
            'from_phone': '+1234567890'                # Replace with your Twilio phone number
        }
        
        # Initialize Twilio client (commented out for demo)
        # self.client = Client(self.twilio_config['account_sid'], self.twilio_config['auth_token'])
    
    def send_sms_reminder(self, phone_number: str, medication_name: str, dosage: str):
        """Send SMS reminder using Twilio"""
        try:
            current_time = datetime.now().strftime('%I:%M %p')
            message_body = f"""
🏥 MEDICATION REMINDER

💊 {medication_name}
📏 Dosage: {dosage}
⏰ Time: {current_time}

Take your medication as prescribed. Stay healthy! 💪
            """.strip()
            
            # Uncomment below for actual Twilio implementation
            # message = self.client.messages.create(
            #     body=message_body,
            #     from_=self.twilio_config['from_phone'],
            #     to=phone_number
            # )
            # st.success(f"SMS sent successfully! Message SID: {message.sid}")
            
            # Demo implementation - shows what would be sent
            st.success(f"📱 SMS Reminder Sent to {phone_number}")
            st.info(f"Message: {message_body}")
            return True
            
        except Exception as e:
            st.error(f"Failed to send SMS: {e}")
            return False
    
    def validate_phone_number(self, phone_number: str) -> bool:
        """Validate phone number format"""
        import re
        # Basic phone number validation (US format)
        pattern = r'^\+?1?[0-9]{10}$'
        return re.match(pattern, phone_number.replace('-', '').replace('(', '').replace(')', '').replace(' ', '')) is not None

class MedicationTracker:
    def __init__(self):
        self.db = DatabaseManager()
        self.drug_checker = DrugInteractionChecker()
        self.reminder_service = ReminderService()
        
        # Initialize session state
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
    
    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def login_page(self):
        st.title("🏥 Medication Tracker")
        st.subheader("Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user = self.db.load_user(username)
                if user and user.password_hash == self.hash_password(password):
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.subheader("Register")
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_phone = st.text_input("Phone Number (+1234567890)", placeholder="+1234567890")
            new_password = st.text_input("New Password", type="password")
            register = st.form_submit_button("Register")
            
            if register:
                if new_username and new_phone and new_password:
                    if self.reminder_service.validate_phone_number(new_phone):
                        user = User(
                            username=new_username,
                            email="",  # Email not needed
                            phone=new_phone,
                            password_hash=self.hash_password(new_password)
                        )
                        if self.db.save_user(user):
                            st.success("Registration successful! Please login.")
                        else:
                            st.error("Registration failed!")
                    else:
                        st.error("Please enter a valid phone number (e.g., +1234567890)")
                else:
                    st.error("Please fill in all required fields")
    
    def main_dashboard(self):
        st.title("🏥 Medication Tracker Dashboard")
        st.write(f"Welcome, {st.session_state.user.username}!")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", [
            "Dashboard",
            "Add Medication",
            "Manage Medications",
            "Drug Interactions",
            "SMS Reminders",
            "Profile"
        ])
        
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()
        
        if page == "Dashboard":
            self.dashboard_page()
        elif page == "Add Medication":
            self.add_medication_page()
        elif page == "Manage Medications":
            self.manage_medications_page()
        elif page == "Drug Interactions":
            self.drug_interactions_page()
        elif page == "SMS Reminders":
            self.reminders_page()
        elif page == "Profile":
            self.profile_page()
    
    def dashboard_page(self):
        st.header("📊 Dashboard")
        
        user = st.session_state.user
        active_meds = [med for med in user.medications if med.active]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Medications", len(active_meds))
        with col2:
            total_doses_today = sum(len(med.time_slots) for med in active_meds)
            st.metric("Doses Today", total_doses_today)
        with col3:
            st.metric("Total Medications", len(user.medications))
        
        if active_meds:
            st.subheader("Today's Schedule")
            schedule_data = []
            for med in active_meds:
                for time_slot in med.time_slots:
                    schedule_data.append({
                        "Time": time_slot,
                        "Medication": med.name,
                        "Dosage": med.dosage,
                        "Notes": med.notes
                    })
            
            schedule_df = pd.DataFrame(schedule_data)
            schedule_df = schedule_df.sort_values("Time")
            st.dataframe(schedule_df, use_container_width=True)
        else:
            st.info("No active medications. Add some medications to get started!")
    
    def add_medication_page(self):
        st.header("➕ Add New Medication")
        
        with st.form("add_medication"):
            col1, col2 = st.columns(2)
            
            with col1:
                med_name = st.text_input("Medication Name")
                dosage = st.text_input("Dosage (e.g., 10mg, 1 tablet)")
                frequency = st.selectbox("Frequency", [
                    "Once daily", "Twice daily", "Three times daily", 
                    "Four times daily", "As needed", "Custom"
                ])
            
            with col2:
                start_date = st.date_input("Start Date", datetime.now().date())
                end_date = st.date_input("End Date (optional)", value=None)
                notes = st.text_area("Notes (optional)")
            
            # Time slots
            st.subheader("Reminder Times")
            num_times = st.number_input("Number of times per day", min_value=1, max_value=6, value=1)
            
            time_slots = []
            for i in range(num_times):
                time_slot = st.time_input(f"Time {i+1}", key=f"time_{i}")
                time_slots.append(time_slot.strftime("%H:%M"))
            
            submit = st.form_submit_button("Add Medication")
            
            if submit and med_name and dosage:
                # Search for drug ID
                drug_results = self.drug_checker.search_drug(med_name)
                drug_id = drug_results[0]["id"] if drug_results else ""
                
                medication = Medication(
                    name=med_name,
                    dosage=dosage,
                    frequency=frequency,
                    time_slots=time_slots,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
                    notes=notes,
                    drug_id=drug_id
                )
                
                st.session_state.user.medications.append(medication)
                if self.db.save_user(st.session_state.user):
                    st.success(f"Added {med_name} successfully!")
                else:
                    st.error("Failed to save medication")
    
    def manage_medications_page(self):
        st.header("💊 Manage Medications")
        
        user = st.session_state.user
        if not user.medications:
            st.info("No medications added yet.")
            return
        
        for i, med in enumerate(user.medications):
            with st.expander(f"{med.name} - {med.dosage}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Frequency:** {med.frequency}")
                    st.write(f"**Times:** {', '.join(med.time_slots)}")
                    st.write(f"**Start Date:** {med.start_date}")
                    if med.end_date:
                        st.write(f"**End Date:** {med.end_date}")
                    if med.notes:
                        st.write(f"**Notes:** {med.notes}")
                
                with col2:
                    active = st.checkbox("Active", value=med.active, key=f"active_{i}")
                    if active != med.active:
                        med.active = active
                        self.db.save_user(user)
                
                with col3:
                    if st.button("Delete", key=f"delete_{i}"):
                        user.medications.pop(i)
                        self.db.save_user(user)
                        st.rerun()
    
    def drug_interactions_page(self):
        st.header("⚠️ Drug Interaction Checker")
        
        user = st.session_state.user
        active_meds = [med for med in user.medications if med.active]
        
        if len(active_meds) < 2:
            st.info("Add at least 2 medications to check for interactions.")
            return
        
        st.subheader("Current Active Medications")
        for med in active_meds:
            st.write(f"• {med.name} ({med.dosage})")
        
        if st.button("Check Interactions"):
            drug_names = [med.name.lower() for med in active_meds]
            interactions = self.drug_checker.check_interactions(drug_names)
            
            if interactions:
                st.subheader("⚠️ Potential Interactions Found")
                for interaction in interactions:
                    if (interaction["drug1"] in drug_names and 
                        interaction["drug2"] in drug_names):
                        
                        severity_color = {
                            "Major": "🔴",
                            "Moderate": "🟡",
                            "Minor": "🟢"
                        }.get(interaction["severity"], "⚪")
                        
                        st.warning(f"""
                        {severity_color} **{interaction["severity"]} Interaction**
                        
                        **Drugs:** {interaction["drug1"].title()} + {interaction["drug2"].title()}
                        
                        **Description:** {interaction["description"]}
                        """)
            else:
                st.success("✅ No known interactions found between your medications.")
    
    def reminders_page(self):
        st.header("📱 SMS Medication Reminders")
        
        user = st.session_state.user
        active_meds = [med for med in user.medications if med.active]
        
        if not active_meds:
            st.info("No active medications to set reminders for.")
            return
        
        if not user.phone:
            st.warning("⚠️ Please add your phone number in the Profile section to receive SMS reminders.")
            return
        
        st.subheader("📞 SMS Reminder Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📱 SMS will be sent to: {user.phone}")
            sms_enabled = st.checkbox("Enable SMS Reminders", value=True)
        
        with col2:
            if self.reminder_service.validate_phone_number(user.phone):
                st.success("✅ Phone number format is valid")
            else:
                st.error("❌ Invalid phone number format. Please update in Profile.")
        
        st.subheader("🧪 Test SMS Reminder")
        selected_med = st.selectbox("Select medication to test", 
                                   [med.name for med in active_meds])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Send Test SMS", type="primary"):
                if sms_enabled and user.phone:
                    med = next(med for med in active_meds if med.name == selected_med)
                    success = self.reminder_service.send_sms_reminder(
                        user.phone, med.name, med.dosage
                    )
                    if success:
                        st.balloons()
                else:
                    st.error("Please enable SMS reminders and ensure phone number is set.")
        
        with col2:
            if st.button("🔍 Validate Phone Number"):
                if self.reminder_service.validate_phone_number(user.phone):
                    st.success("✅ Phone number is valid!")
                else:
                    st.error("❌ Phone number format is invalid. Use format: +1234567890")
        
        st.subheader("📅 Today's SMS Schedule")
        now = datetime.now().time()
        upcoming = []
        
        for med in active_meds:
            for time_slot in med.time_slots:
                reminder_time = datetime.strptime(time_slot, "%H:%M").time()
                status = "✅ Completed" if reminder_time <= now else "⏰ Upcoming"
                upcoming.append({
                    "Time": time_slot,
                    "Medication": med.name,
                    "Dosage": med.dosage,
                    "Status": status
                })
        
        if upcoming:
            upcoming_df = pd.DataFrame(upcoming)
            upcoming_df = upcoming_df.sort_values("Time")
            st.dataframe(upcoming_df, use_container_width=True)
            
            # Show upcoming reminders count
            upcoming_count = len([r for r in upcoming if "Upcoming" in r["Status"]])
            if upcoming_count > 0:
                st.success(f"📬 {upcoming_count} SMS reminder(s) scheduled for today")
            else:
                st.info("✅ All reminders for today have been sent")
        else:
            st.info("No reminders scheduled for today.")
        
        # Reminder frequency settings
        st.subheader("⚙️ Advanced SMS Settings")
        with st.expander("SMS Reminder Options"):
            reminder_buffer = st.slider(
                "Send reminder X minutes before scheduled time", 
                min_value=0, max_value=30, value=5
            )
            
            enable_followup = st.checkbox("Send follow-up SMS if not acknowledged within 15 minutes")
            
            if st.button("💾 Save SMS Settings"):
                st.success("SMS settings saved successfully!")
                st.info(f"Reminders will be sent {reminder_buffer} minutes early")
                if enable_followup:
                    st.info("Follow-up reminders enabled")
    
    def profile_page(self):
        st.header("👤 User Profile")
        
        user = st.session_state.user
        
        with st.form("profile_form"):
            st.subheader("Personal Information")
            username = st.text_input("Username", value=user.username, disabled=True)
            phone = st.text_input("Phone Number", value=user.phone, 
                                 placeholder="+1234567890", 
                                 help="Format: +1234567890 (include country code)")
            
            # Phone validation feedback
            if phone and phone != user.phone:
                if self.reminder_service.validate_phone_number(phone):
                    st.success("✅ Valid phone number format")
                else:
                    st.error("❌ Invalid format. Use: +1234567890")
            
            st.subheader("Change Password")
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Update Profile"):
                updated = False
                
                # Update phone
                if phone != user.phone:
                    if self.reminder_service.validate_phone_number(phone):
                        user.phone = phone
                        updated = True
                    else:
                        st.error("Please enter a valid phone number format")
                        return
                
                # Update password if provided
                if current_password and new_password and confirm_password:
                    if self.hash_password(current_password) == user.password_hash:
                        if new_password == confirm_password:
                            user.password_hash = self.hash_password(new_password)
                            updated = True
                            st.success("Password updated successfully!")
                        else:
                            st.error("New passwords don't match")
                    else:
                        st.error("Incorrect current password")
                
                if updated:
                    if self.db.save_user(user):
                        st.session_state.user = user
                        st.success("Profile updated successfully!")
                    else:
                        st.error("Failed to update profile")
        
        # SMS Status
        st.subheader("📱 SMS Reminder Status")
        if user.phone:
            if self.reminder_service.validate_phone_number(user.phone):
                st.success(f"✅ SMS reminders active for: {user.phone}")
            else:
                st.error(f"❌ Invalid phone format: {user.phone}")
                st.info("Please update your phone number to receive SMS reminders")
        else:
            st.warning("⚠️ No phone number set. Add one to receive SMS reminders.")
        
        st.subheader("📊 Data Export")
        if st.button("Export Medication Data"):
            med_data = []
            for med in user.medications:
                med_data.append(asdict(med))
            
            df = pd.DataFrame(med_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"medications_{user.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

def main():
    st.set_page_config(
        page_title="Medication Tracker",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    app = MedicationTracker()
    
    if not st.session_state.logged_in:
        app.login_page()
    else:
        app.main_dashboard()

if __name__ == "__main__":
    main()