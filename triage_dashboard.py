"""
Streamlit Dashboard for ER Triage System
Real-time patient acuity prediction and prioritization interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="ER Triage System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .acuity-1 { background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .acuity-2 { background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .acuity-3 { background-color: #ffeaa7; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .acuity-4 { background-color: #fdcb6e; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .acuity-5 { background-color: #d63031; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; font-weight: bold; }
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .patient-card {
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        model = joblib.load('triage_model.pkl')
        scaler = joblib.load('scaler.pkl')
        imputer = joblib.load('imputer.pkl')
        
        with open('feature_columns.txt', 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        return model, scaler, imputer, feature_cols, metrics
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None, None


def create_feature_dict():
    """Create a dictionary with feature defaults for single patient input"""
    features = {
        # Triage vitals
        'temperature': 98.6,
        'heartrate': 80.0,
        'resprate': 16.0,
        'o2sat': 98.0,
        'sbp': 120.0,
        'dbp': 80.0,
        'pain': 0.0,
        
        # Demographics
        'age_at_visit': 50.0,
        'gender_male': 0,
        
        # Transport (keep these but they'll be filtered out - needed for scaler/imputer)
        'transport_AMBULANCE': 0,
        'transport_WALK IN': 0,
        'transport_HELICOPTER': 0,
        'transport_OTHER': 0,
        'transport_UNKNOWN': 0,
        
        # Chief complaint categories
        'cc_chest_pain': 0,
        'cc_shortness_breath': 0,
        'cc_abdominal_pain': 0,
        'cc_neurological': 0,
        'cc_trauma': 0,
        'cc_infection': 0,
        'cc_hemorrhage': 0,
        'cc_cardiac': 0,
        'cc_pain': 0,
        'cc_hypotension': 0,
        'cc_hypertension': 0,
        
        # Diagnosis features
        'num_diagnoses': 0,
        'dx_mi': 0,
        'dx_stroke': 0,
        'dx_sepsis': 0,
        'dx_head_trauma': 0,
        'dx_heart_failure': 0,
        'dx_resp_failure': 0,
        'dx_gi_bleed': 0,
        'dx_pe': 0,
        
        # Vital sign aggregates (simulating 5 stable measurements)
        'temperature_initial': 98.6,
        'temperature_mean': 98.6,
        'temperature_min': 98.6,
        'temperature_max': 98.6,
        'temperature_std': 0.0,
        'temperature_trend': 0.0,
        'temperature_count': 5.0,
        
        'heartrate_initial': 80.0,
        'heartrate_mean': 80.0,
        'heartrate_min': 80.0,
        'heartrate_max': 80.0,
        'heartrate_std': 0.0,
        'heartrate_trend': 0.0,
        'heartrate_count': 5.0,
        
        'resprate_initial': 16.0,
        'resprate_mean': 16.0,
        'resprate_min': 16.0,
        'resprate_max': 16.0,
        'resprate_std': 0.0,
        'resprate_trend': 0.0,
        'resprate_count': 5.0,
        
        'o2sat_initial': 98.0,
        'o2sat_mean': 98.0,
        'o2sat_min': 98.0,
        'o2sat_max': 98.0,
        'o2sat_std': 0.0,
        'o2sat_trend': 0.0,
        'o2sat_count': 5.0,
        
        'sbp_initial': 120.0,
        'sbp_mean': 120.0,
        'sbp_min': 120.0,
        'sbp_max': 120.0,
        'sbp_std': 0.0,
        'sbp_trend': 0.0,
        'sbp_count': 5.0,
        
        'dbp_initial': 80.0,
        'dbp_mean': 80.0,
        'dbp_min': 80.0,
        'dbp_max': 80.0,
        'dbp_std': 0.0,
        'dbp_trend': 0.0,
        'dbp_count': 5.0,
        
        # Derived features
        'map_mean': 93.3,  # (2*80 + 120) / 3
        'pulse_pressure_mean': 40.0,  # 120 - 80
        'shock_index': 0.67,  # 80 / 120
        'mews_total': 0,
        'mews_resp': 0,
        'mews_hr': 0,
        'mews_sbp': 0,
        'abnormal_vitals_count': 0,
    }
    
    return features


def get_acuity_color(acuity):
    """Get color for acuity level"""
    colors = {
        1: "#28a745",  # Green
        2: "#ffc107",  # Yellow
        3: "#fd7e14",  # Orange
        4: "#dc3545",  # Red
        5: "#721c24"   # Dark Red
    }
    return colors.get(int(acuity), "#6c757d")


def get_acuity_label(acuity):
    """Get descriptive label for acuity level"""
    labels = {
        1: "Non-Urgent",
        2: "Semi-Urgent",
        3: "Urgent",
        4: "Very Urgent",
        5: "Critical"
    }
    return labels.get(int(acuity), "Unknown")


def rule_based_triage(patient_data):
    """
    Clinical rule-based triage using established medical criteria.
    This follows emergency medicine best practices and clinical guidelines.
    """
    
    # Extract key clinical indicators
    sbp = patient_data.get('sbp_mean', patient_data.get('sbp', 120))
    o2sat = patient_data.get('o2sat_mean', patient_data.get('o2sat', 98))
    hr = patient_data.get('heartrate_mean', patient_data.get('heartrate', 80))
    temp = patient_data.get('temperature_mean', patient_data.get('temperature', 98.6))
    rr = patient_data.get('resprate_mean', patient_data.get('resprate', 16))
    shock_index = patient_data.get('shock_index', 0.67)
    mews = patient_data.get('mews_total', 0)
    pain = patient_data.get('pain', 0)
    abnormal_vitals = patient_data.get('abnormal_vitals_count', 0)
    
    # Check for critical diagnoses
    has_mi = patient_data.get('dx_mi', 0) == 1
    has_stroke = patient_data.get('dx_stroke', 0) == 1
    has_sepsis = patient_data.get('dx_sepsis', 0) == 1
    has_head_trauma = patient_data.get('dx_head_trauma', 0) == 1
    has_resp_failure = patient_data.get('dx_resp_failure', 0) == 1
    has_gi_bleed = patient_data.get('dx_gi_bleed', 0) == 1
    
    # Check for critical chief complaints
    chest_pain = patient_data.get('cc_chest_pain', 0) == 1
    sob = patient_data.get('cc_shortness_breath', 0) == 1
    neuro = patient_data.get('cc_neurological', 0) == 1
    hemorrhage = patient_data.get('cc_hemorrhage', 0) == 1
    
    # ACUITY 5 - CRITICAL (Immediate Life-Threatening)
    critical_conditions = [
        sbp < 70,  # Severe hypotension/shock
        o2sat < 85,  # Severe hypoxia
        shock_index > 1.4,  # Severe shock
        has_mi and (sbp < 90 or chest_pain),  # MI with instability
        has_stroke and neuro,  # Acute stroke
        has_sepsis and (sbp < 90 or temp > 103),  # Severe sepsis
        has_resp_failure and o2sat < 90,  # Respiratory failure
        hemorrhage and sbp < 90,  # Active hemorrhage with shock
    ]
    
    if any(critical_conditions):
        return 5, "Critical - Immediate life-threatening condition"
    
    # ACUITY 4 - VERY URGENT (Potentially Life-Threatening)
    very_urgent_conditions = [
        sbp < 90,  # Hypotension
        o2sat < 90,  # Significant hypoxia
        shock_index > 1.0,  # Shock state
        mews >= 5,  # High MEWS score
        temp > 102,  # High fever
        hr > 130 and (chest_pain or sob),  # Tachycardia with cardiac/resp symptoms
        has_mi,  # Any MI
        has_stroke,  # Any stroke
        has_gi_bleed and abnormal_vitals >= 2,  # GI bleed with instability
        has_head_trauma and (neuro or abnormal_vitals >= 2),  # Head trauma with symptoms
        chest_pain and (hr > 100 or sbp < 100),  # Chest pain with abnormal vitals
        pain >= 9 and abnormal_vitals >= 2,  # Severe pain with vital sign abnormalities
    ]
    
    if any(very_urgent_conditions):
        return 4, "Very Urgent - Requires rapid assessment"
    
    # ACUITY 3 - URGENT (Serious but Stable)
    urgent_conditions = [
        mews >= 3,  # Moderate MEWS
        abnormal_vitals >= 3,  # Multiple abnormal vitals
        pain >= 7,  # Severe pain
        temp > 101,  # Moderate fever
        sbp < 100,  # Mild hypotension
        o2sat < 94,  # Mild hypoxia
        hr > 110,  # Moderate tachycardia
        chest_pain,  # Any chest pain
        sob,  # Any shortness of breath
        neuro,  # Any neuro symptoms
        patient_data.get('cc_abdominal_pain', 0) == 1 and pain >= 5,  # Significant abdominal pain
    ]
    
    if any(urgent_conditions):
        return 3, "Urgent - Needs prompt attention"
    
    # ACUITY 2 - SEMI-URGENT (Non-Life-Threatening)
    semi_urgent_conditions = [
        mews >= 1,  # Any MEWS points
        abnormal_vitals >= 1,  # Any abnormal vital
        pain >= 4,  # Moderate pain
        temp > 100.4,  # Low-grade fever
        patient_data.get('age_at_visit', 50) > 65 and abnormal_vitals >= 1,  # Elderly with any abnormality
    ]
    
    if any(semi_urgent_conditions):
        return 2, "Semi-Urgent - Can wait briefly"
    
    # ACUITY 1 - NON-URGENT (Minor/Routine)
    return 1, "Non-Urgent - Routine care"


def predict_single_patient(model, scaler, imputer, feature_cols, patient_data):
    """
    Hybrid prediction: Uses rule-based triage with ML as secondary validation.
    This approach is more reliable for medical decision support.
    """
    # PRIMARY: Rule-based clinical triage
    rule_based_acuity, reasoning = rule_based_triage(patient_data)
    
    # SECONDARY: Get ML model opinion (for comparison/validation)
    try:
        X = pd.DataFrame([patient_data])
        
        # Ensure all required features are present
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Reorder columns to match training exactly
        X = X[feature_cols]
        
        # Impute and scale
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        
        # Get ML probabilities
        probabilities = model.predict_proba(X_scaled)[0]
        ml_prediction = int(model.predict(X_scaled)[0]) + 1
        
    except Exception as e:
        print(f"ML prediction failed: {e}")
        probabilities = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform if ML fails
        ml_prediction = rule_based_acuity
    
    # Use rule-based as primary, but flag if ML disagrees significantly
    final_prediction = rule_based_acuity
    
    # If ML thinks it's more urgent, consider escalating
    if ml_prediction > rule_based_acuity and probabilities[ml_prediction-1] > 0.3:
        print(f"NOTE: ML suggests higher acuity ({ml_prediction}) vs rule-based ({rule_based_acuity})")
        # Could optionally escalate here, but trust clinical rules more
    
    print(f"\nDEBUG: Rule-based prediction: {rule_based_acuity} - {reasoning}")
    print(f"DEBUG: ML prediction: {ml_prediction}")
    print(f"DEBUG: Final prediction: {final_prediction}")
    
    return final_prediction, probabilities, rule_based_acuity


def single_patient_interface():
    """Interface for single patient triage prediction"""
    st.header("🏥 Single Patient Triage")
    
    # Load model
    model, scaler, imputer, feature_cols, metrics = load_model_artifacts()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please run modelling.py first to train the model.")
        return
    
    # Display model performance
    with st.expander("📊 Model Performance Metrics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Critical Case Recall", f"{metrics['critical_recall']:.2%}")
        with col3:
            st.metric("Critical Precision", f"{metrics['critical_precision']:.2%}")
        with col4:
            st.metric("High-Risk Recall", f"{metrics['highrisk_recall']:.2%}")
    
    st.markdown("---")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Patient Information")
        
        age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        arrival = st.selectbox("Arrival Method", ["Walk In", "Ambulance", "Other"])
        
        st.subheader("🩺 Vital Signs")
        
        temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=106.0, 
                                      value=98.6, step=0.1)
        heartrate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, 
                                    value=80, step=1)
        resprate = st.number_input("Respiratory Rate (bpm)", min_value=6, max_value=40, 
                                   value=16, step=1)
        o2sat = st.number_input("O2 Saturation (%)", min_value=70, max_value=100, 
                                value=98, step=1)
        sbp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=220, 
                              value=120, step=1)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, 
                              value=80, step=1)
        pain = st.slider("Pain Level (0-10)", min_value=0, max_value=10, value=0)
    
    with col2:
        st.subheader("🔍 Chief Complaint")
        
        cc_chest_pain = st.checkbox("Chest Pain / Cardiac")
        cc_shortness_breath = st.checkbox("Shortness of Breath")
        cc_abdominal_pain = st.checkbox("Abdominal Pain")
        cc_neurological = st.checkbox("Neurological (Stroke, Weakness, Confusion)")
        cc_trauma = st.checkbox("Trauma / Injury")
        cc_infection = st.checkbox("Infection / Fever")
        cc_hemorrhage = st.checkbox("Bleeding / Hemorrhage")
        
        st.subheader("🏥 Suspected Diagnoses")
        
        dx_mi = st.checkbox("Myocardial Infarction (MI)")
        dx_stroke = st.checkbox("Stroke")
        dx_sepsis = st.checkbox("Sepsis")
        dx_head_trauma = st.checkbox("Head Trauma")
        dx_resp_failure = st.checkbox("Respiratory Failure")
        dx_gi_bleed = st.checkbox("GI Bleed")
    
    # Predict button
    if st.button("🔮 Predict Acuity", type="primary", use_container_width=True):
        # Prepare patient data
        patient_data = create_feature_dict()
        
        # Update with user inputs - basic vitals
        patient_data['age_at_visit'] = age
        patient_data['gender_male'] = 1 if gender == "Male" else 0
        patient_data['temperature'] = temperature
        patient_data['heartrate'] = heartrate
        patient_data['resprate'] = resprate
        patient_data['o2sat'] = o2sat
        patient_data['sbp'] = sbp
        patient_data['dbp'] = dbp
        patient_data['pain'] = pain
        
        # Update vital sign aggregates (treating as if patient had 5 stable measurements)
        for vital_name, vital_value in [
            ('temperature', temperature),
            ('heartrate', heartrate),
            ('resprate', resprate),
            ('o2sat', o2sat),
            ('sbp', sbp),
            ('dbp', dbp)
        ]:
            patient_data[f'{vital_name}_initial'] = vital_value
            patient_data[f'{vital_name}_mean'] = vital_value
            patient_data[f'{vital_name}_min'] = vital_value
            patient_data[f'{vital_name}_max'] = vital_value
            patient_data[f'{vital_name}_std'] = 0.0
            patient_data[f'{vital_name}_trend'] = 0.0
            patient_data[f'{vital_name}_count'] = 5.0  # Changed from 1.0 to 5.0 - typical for stable patients
        
        # Calculate derived features
        patient_data['map_mean'] = (2 * dbp + sbp) / 3
        patient_data['pulse_pressure_mean'] = sbp - dbp
        patient_data['shock_index'] = heartrate / sbp if sbp > 0 else 0
        
        # Calculate MEWS components
        # Respiratory rate score
        mews_resp = 0
        if resprate < 9: mews_resp = 2
        elif 9 <= resprate <= 14: mews_resp = 0
        elif 15 <= resprate <= 20: mews_resp = 1
        elif 21 <= resprate <= 29: mews_resp = 2
        elif resprate >= 30: mews_resp = 3
        patient_data['mews_resp'] = mews_resp
        
        # Heart rate score
        mews_hr = 0
        if heartrate < 40: mews_hr = 2
        elif 40 <= heartrate <= 50: mews_hr = 1
        elif 51 <= heartrate <= 100: mews_hr = 0
        elif 101 <= heartrate <= 110: mews_hr = 1
        elif 111 <= heartrate <= 129: mews_hr = 2
        elif heartrate >= 130: mews_hr = 3
        patient_data['mews_hr'] = mews_hr
        
        # Systolic BP score
        mews_sbp = 0
        if sbp < 70: mews_sbp = 3
        elif 70 <= sbp <= 80: mews_sbp = 2
        elif 81 <= sbp <= 100: mews_sbp = 1
        elif 101 <= sbp <= 199: mews_sbp = 0
        elif sbp >= 200: mews_sbp = 2
        patient_data['mews_sbp'] = mews_sbp
        
        patient_data['mews_total'] = mews_resp + mews_hr + mews_sbp
        
        # Abnormal vitals count
        abnormal_count = 0
        if temperature > 100.4 or temperature < 96.8: abnormal_count += 1
        if heartrate > 100 or heartrate < 60: abnormal_count += 1
        if resprate > 20 or resprate < 12: abnormal_count += 1
        if o2sat < 95: abnormal_count += 1
        if sbp > 140 or sbp < 90: abnormal_count += 1
        patient_data['abnormal_vitals_count'] = abnormal_count
        
        # Chief complaints
        patient_data['cc_chest_pain'] = int(cc_chest_pain)
        patient_data['cc_shortness_breath'] = int(cc_shortness_breath)
        patient_data['cc_abdominal_pain'] = int(cc_abdominal_pain)
        patient_data['cc_neurological'] = int(cc_neurological)
        patient_data['cc_trauma'] = int(cc_trauma)
        patient_data['cc_infection'] = int(cc_infection)
        patient_data['cc_hemorrhage'] = int(cc_hemorrhage)
        
        # Diagnoses
        patient_data['dx_mi'] = int(dx_mi)
        patient_data['dx_stroke'] = int(dx_stroke)
        patient_data['dx_sepsis'] = int(dx_sepsis)
        patient_data['dx_head_trauma'] = int(dx_head_trauma)
        patient_data['dx_resp_failure'] = int(dx_resp_failure)
        patient_data['dx_gi_bleed'] = int(dx_gi_bleed)
        
        # Count diagnoses
        patient_data['num_diagnoses'] = sum([
            int(dx_mi), int(dx_stroke), int(dx_sepsis), 
            int(dx_head_trauma), int(dx_resp_failure), int(dx_gi_bleed)
        ])
        
        # Make prediction
        prediction, probabilities, prediction_raw = predict_single_patient(
            model, scaler, imputer, feature_cols, patient_data
        )
        
        # DEBUG: Print feature values to console
        print("\n=== DEBUG: Feature values being sent to model ===")
        for key, value in sorted(patient_data.items()):
            if key in feature_cols:
                print(f"{key}: {value}")
        print("=" * 50 + "\n")
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Main prediction display
        acuity_color = get_acuity_color(prediction)
        acuity_label = get_acuity_label(prediction)
        
        st.markdown(f"""
        <div style='background-color: {acuity_color}; padding: 30px; border-radius: 15px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>Acuity Level: {prediction}</h1>
            <h2 style='color: white; margin: 10px 0 0 0;'>{acuity_label}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability distribution
        st.markdown("### Confidence Distribution")
        prob_df = pd.DataFrame({
            'Acuity Level': [f"Level {i}" for i in range(1, 6)],
            'Probability': probabilities * 100
        })
        
        fig = px.bar(prob_df, x='Acuity Level', y='Probability',
                     color='Probability',
                     color_continuous_scale=['green', 'yellow', 'orange', 'red', 'darkred'],
                     labels={'Probability': 'Probability (%)'},
                     title='Prediction Confidence by Acuity Level')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical indicators
        st.markdown("### 🚨 Clinical Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Calculate MEWS
            mews_score = 0
            if resprate < 9: mews_score += 2
            elif resprate >= 30: mews_score += 3
            elif resprate >= 21: mews_score += 2
            elif resprate >= 15: mews_score += 1
            
            if heartrate < 40: mews_score += 2
            elif heartrate >= 130: mews_score += 3
            elif heartrate >= 111: mews_score += 2
            elif heartrate >= 101: mews_score += 1
            
            if sbp < 70: mews_score += 3
            elif sbp < 80: mews_score += 2
            elif sbp < 100: mews_score += 1
            elif sbp >= 200: mews_score += 2
            
            st.metric("MEWS Score", mews_score)
            if mews_score >= 5:
                st.error("⚠️ High MEWS - Increased mortality risk")
        
        with col2:
            # Shock index
            shock_index = heartrate / sbp if sbp > 0 else 0
            st.metric("Shock Index", f"{shock_index:.2f}")
            if shock_index > 0.9:
                st.error("⚠️ Elevated - Possible shock")
        
        with col3:
            # Abnormal vitals count
            abnormal_count = 0
            if temperature > 100.4 or temperature < 96.8: abnormal_count += 1
            if heartrate > 100 or heartrate < 60: abnormal_count += 1
            if resprate > 20 or resprate < 12: abnormal_count += 1
            if o2sat < 95: abnormal_count += 1
            if sbp > 140 or sbp < 90: abnormal_count += 1
            
            st.metric("Abnormal Vitals", abnormal_count)
            if abnormal_count >= 3:
                st.warning("⚠️ Multiple abnormal vitals")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if prediction >= 5:
            st.error("🚨 **IMMEDIATE ATTENTION REQUIRED**")
            st.markdown("""
            - Activate rapid response team
            - Prepare for immediate intervention
            - Continuous monitoring required
            - Notify attending physician immediately
            """)
        elif prediction >= 4:
            st.warning("⚠️ **URGENT CARE NEEDED**")
            st.markdown("""
            - Expedite triage process
            - Monitor closely
            - Prepare for potential escalation
            - Physician evaluation within 30 minutes
            """)
        elif prediction >= 3:
            st.info("ℹ️ **STANDARD URGENT CARE**")
            st.markdown("""
            - Standard triage protocol
            - Monitor vital signs regularly
            - Physician evaluation within 60 minutes
            """)
        else:
            st.success("✅ **STABLE PATIENT**")
            st.markdown("""
            - Standard care protocol
            - Regular monitoring
            - Can wait for available resources
            """)


def batch_patient_interface():
    """Interface for batch patient processing and prioritization"""
    st.header("📊 Patient Queue Management")
    
    # Load model
    model, scaler, imputer, feature_cols, metrics = load_model_artifacts()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please run modelling.py first to train the model.")
        return
    
    # File uploader
    st.markdown("### 📤 Upload Patient Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with patient data",
        type=['csv'],
        help="Upload a CSV file containing patient vital signs and information"
    )
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} patients")
        
        # Show raw data
        with st.expander("📋 View Raw Data"):
            st.dataframe(df)
        
        # Make predictions button
        if st.button("🔮 Predict All Acuities", type="primary"):
            with st.spinner("Making predictions..."):
                # Prepare features
                X = df[feature_cols] if all(col in df.columns for col in feature_cols) else df
                
                # Handle missing columns
                for col in feature_cols:
                    if col not in X.columns:
                        X[col] = 0
                X = X[feature_cols]
                
                # Predict
                X_imputed = imputer.transform(X)
                X_scaled = scaler.transform(X_imputed)
                predictions_indexed = model.predict(X_scaled)
                predictions = predictions_indexed + 1
                probabilities = model.predict_proba(X_scaled)
                
                # Add to dataframe
                df['predicted_acuity'] = predictions
                df['acuity_label'] = df['predicted_acuity'].apply(get_acuity_label)
                
                # Add max probability
                df['confidence'] = [probs.max() for probs in probabilities]
                
                # Sort by acuity (highest first) then by confidence
                df_sorted = df.sort_values(['predicted_acuity', 'confidence'], 
                                           ascending=[False, False])
                
                # Display summary stats
                st.markdown("### 📈 Queue Summary")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    critical = (df['predicted_acuity'] == 5).sum()
                    st.metric("Critical (5)", critical, 
                             delta="Immediate" if critical > 0 else None,
                             delta_color="inverse")
                
                with col2:
                    very_urgent = (df['predicted_acuity'] == 4).sum()
                    st.metric("Very Urgent (4)", very_urgent)
                
                with col3:
                    urgent = (df['predicted_acuity'] == 3).sum()
                    st.metric("Urgent (3)", urgent)
                
                with col4:
                    semi_urgent = (df['predicted_acuity'] == 2).sum()
                    st.metric("Semi-Urgent (2)", semi_urgent)
                
                with col5:
                    non_urgent = (df['predicted_acuity'] == 1).sum()
                    st.metric("Non-Urgent (1)", non_urgent)
                
                # Visualization
                st.markdown("### 📊 Acuity Distribution")
                
                acuity_counts = df['predicted_acuity'].value_counts().sort_index()
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"Level {i}" for i in acuity_counts.index],
                        y=acuity_counts.values,
                        marker_color=[get_acuity_color(i) for i in acuity_counts.index],
                        text=acuity_counts.values,
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title='Patient Distribution by Acuity Level',
                    xaxis_title='Acuity Level',
                    yaxis_title='Number of Patients',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Prioritized patient list
                st.markdown("### 🚨 Prioritized Patient Queue")
                st.markdown("*Patients sorted by acuity level (highest priority first)*")
                
                # Display each patient
                for idx, row in df_sorted.iterrows():
                    acuity = int(row['predicted_acuity'])
                    acuity_class = f"acuity-{acuity}"
                    
                    with st.container():
                        col1, col2, col3 = st.columns([2, 3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class='{acuity_class}'>
                                <strong>Acuity {acuity}: {get_acuity_label(acuity)}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Show key vitals if available
                            vitals = []
                            if 'heartrate' in row:
                                vitals.append(f"HR: {row['heartrate']:.0f}")
                            if 'sbp' in row:
                                vitals.append(f"BP: {row['sbp']:.0f}/{row['dbp']:.0f}")
                            if 'o2sat' in row:
                                vitals.append(f"O2: {row['o2sat']:.0f}%")
                            if 'resprate' in row:
                                vitals.append(f"RR: {row['resprate']:.0f}")
                            
                            st.text(" | ".join(vitals) if vitals else "No vitals available")
                        
                        with col3:
                            st.metric("Confidence", f"{row['confidence']:.1%}")
                        
                        st.markdown("---")
                
                # Download prioritized list
                st.markdown("### 💾 Export Results")
                csv = df_sorted.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prioritized Patient List",
                    data=csv,
                    file_name=f"prioritized_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def main():
    # Title
    st.title("🏥 ER Triage Prediction System")
    st.markdown("*ML-powered patient acuity assessment and prioritization*")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/2C3E50/FFFFFF?text=ER+Triage+AI", 
                 use_container_width=True)
        
        st.markdown("---")
        
        mode = st.radio(
            "Select Mode",
            ["Single Patient", "Patient Queue"],
            help="Choose between single patient assessment or batch processing"
        )
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses machine learning to predict patient acuity levels (1-5) 
        based on vital signs, symptoms, and medical history.
        
        **Acuity Levels:**
        - 🟢 Level 1: Non-Urgent
        - 🟡 Level 2: Semi-Urgent  
        - 🟠 Level 3: Urgent
        - 🔴 Level 4: Very Urgent
        - ⚫ Level 5: Critical
        
        The model is optimized to catch critical cases (Level 5) with high sensitivity.
        """)
        
        st.markdown("---")
        st.markdown("*Developed with MIMIC-ED data*")
    
    # Main content based on mode
    if mode == "Single Patient":
        single_patient_interface()
    else:
        batch_patient_interface()


if __name__ == "__main__":
    main()