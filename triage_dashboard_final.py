"""
ER Triage System - Final Dashboard
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
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .acuity-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 10px 0;
    }
    .level-5 { background: linear-gradient(135deg, #8E44AD 0%, #9B59B6 100%); }
    .level-4 { background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%); }
    .level-3 { background: linear-gradient(135deg, #F39C12 0%, #E67E22 100%); }
    .level-2 { background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%); }
    .level-1 { background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%); }
</style>
""", unsafe_allow_html=True)

# Session state
if 'patient_queue' not in st.session_state:
    st.session_state.patient_queue = []
if 'patient_counter' not in st.session_state:
    st.session_state.patient_counter = 1

# Load model
@st.cache_resource
def load_model_artifacts():
    """Load model artifacts"""
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
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

def rule_based_triage(patient_data):
    """Clinical rule-based triage"""
    sbp = patient_data.get('sbp', 120)
    o2sat = patient_data.get('o2sat', 98)
    hr = patient_data.get('heartrate', 80)
    temp = patient_data.get('temperature', 98.6)
    shock_index = patient_data.get('shock_index', 0.67)
    mews = patient_data.get('mews_total', 0)
    pain = patient_data.get('pain', 0)
    abnormal_vitals = patient_data.get('abnormal_vitals_count', 0)
    
    # Critical diagnoses
    has_mi = patient_data.get('dx_mi', 0) == 1
    has_stroke = patient_data.get('dx_stroke', 0) == 1
    has_sepsis = patient_data.get('dx_sepsis', 0) == 1
    
    # Chief complaints
    chest_pain = patient_data.get('cc_chest_pain', 0) == 1
    sob = patient_data.get('cc_shortness_breath', 0) == 1
    neuro = patient_data.get('cc_neurological', 0) == 1
    
    # ACUITY 5 - CRITICAL
    if (sbp < 70 or o2sat < 85 or shock_index > 1.4 or 
        (has_mi and sbp < 90) or (has_stroke and neuro) or 
        (has_sepsis and (sbp < 90 or temp > 103))):
        return 5, "Critical - Immediate life threat"
    
    # ACUITY 4 - VERY URGENT
    if (sbp < 90 or o2sat < 90 or shock_index > 1.0 or mews >= 5 or 
        temp > 102 or has_mi or has_stroke or 
        (chest_pain and (hr > 100 or sbp < 100))):
        return 4, "Very Urgent - Rapid assessment needed"
    
    # ACUITY 3 - URGENT
    if (mews >= 3 or abnormal_vitals >= 3 or pain >= 7 or temp > 101 or 
        sbp < 100 or o2sat < 94 or chest_pain or sob):
        return 3, "Urgent - Prompt attention needed"
    
    # ACUITY 2 - SEMI-URGENT
    if (mews >= 1 or abnormal_vitals >= 1 or pain >= 4 or temp > 100.4):
        return 2, "Semi-Urgent - Can wait briefly"
    
    # ACUITY 1 - NON-URGENT
    return 1, "Non-Urgent - Routine care"

def predict_patient(patient_data, model, scaler, imputer, feature_cols):
    """Predict using hybrid system"""
    # Rule-based prediction
    acuity, reasoning = rule_based_triage(patient_data)
    
    # ML for confidence/validation (optional)
    try:
        X = pd.DataFrame([patient_data])
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols]
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        probabilities = model.predict_proba(X_scaled)[0]
    except:
        probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    return acuity, reasoning, probabilities

# Sidebar - Add Patient
def render_sidebar(model, scaler, imputer, feature_cols):
    st.sidebar.title("➕ Add New Patient")
    
    with st.sidebar.form("patient_form"):
        st.subheader("Patient Information")
        name = st.text_input("Name/ID", f"Patient-{st.session_state.patient_counter}")
        age = st.number_input("Age", 0, 120, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        st.subheader("Vital Signs")
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("Temperature (°F)", 95.0, 108.0, 98.6, 0.1)
            hr = st.number_input("Heart Rate (bpm)", 30, 200, 80)
            rr = st.number_input("Respiratory Rate", 8, 50, 16)
            spo2 = st.number_input("O2 Saturation (%)", 70, 100, 98)
        with col2:
            sbp = st.number_input("Systolic BP", 60, 250, 120)
            dbp = st.number_input("Diastolic BP", 40, 150, 80)
            pain = st.slider("Pain Level (0-10)", 0, 10, 0)
        
        st.subheader("Chief Complaint")
        col1, col2 = st.columns(2)
        with col1:
            cc_chest = st.checkbox("Chest Pain / Cardiac")
            cc_sob = st.checkbox("Shortness of Breath")
            cc_abd = st.checkbox("Abdominal Pain")
            cc_neuro = st.checkbox("Neurological (Stroke, Weakness, Confusion)")
        with col2:
            cc_trauma = st.checkbox("Trauma / Injury")
            cc_infection = st.checkbox("Infection / Fever")
            cc_hemorrhage = st.checkbox("Bleeding / Hemorrhage")
        
        st.subheader("Suspected Diagnoses")
        col1, col2 = st.columns(2)
        with col1:
            dx_mi = st.checkbox("MI (Myocardial Infarction)")
            dx_stroke = st.checkbox("Stroke / CVA")
            dx_sepsis = st.checkbox("Sepsis")
            dx_head = st.checkbox("Head Trauma")
        with col2:
            dx_resp = st.checkbox("Respiratory Failure")
            dx_heart_failure = st.checkbox("Heart Failure")
            dx_gi_bleed = st.checkbox("GI Bleed")
            dx_pe = st.checkbox("Pulmonary Embolism")
        
        st.subheader("Past Medical History / Comorbidities")
        with st.expander("📋 Click to add comorbidities (ICD categories)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Cardiovascular**")
                pmh_cardiac = st.checkbox("Cardiac Disease")
                pmh_htn = st.checkbox("Hypertension")
                
                st.write("**Respiratory**")
                pmh_respiratory = st.checkbox("Chronic Respiratory Disease")
                
                st.write("**Metabolic/Endocrine**")
                pmh_diabetes = st.checkbox("Diabetes")
                pmh_endocrine = st.checkbox("Other Endocrine Disorder")
                
                st.write("**Renal/GU**")
                pmh_renal = st.checkbox("Kidney Disease")
                pmh_gu = st.checkbox("Genitourinary Issues")
                
            with col2:
                st.write("**Hematologic**")
                pmh_blood = st.checkbox("Blood Disorders/Anemia")
                
                st.write("**Infectious**")
                pmh_infectious = st.checkbox("Chronic Infections")
                
                st.write("**Musculoskeletal**")
                pmh_msk = st.checkbox("MSK Disorders")
                
                st.write("**Neurologic**")
                pmh_neuro = st.checkbox("Neurologic Conditions")
                
                st.write("**Other**")
                pmh_cancer = st.checkbox("Cancer/Malignancy")
                pmh_symptoms = st.checkbox("Chronic Symptoms")
        
        submit = st.form_submit_button("🚑 Add to Queue", use_container_width=True)
        
        if submit:
            # Calculate derived vitals
            map_val = (2 * dbp + sbp) / 3
            shock_idx = hr / sbp if sbp > 0 else 0
            
            # MEWS calculation
            mews_resp = 1 if 15 <= rr <= 20 else (2 if 21 <= rr <= 29 else 0)
            mews_hr = 0 if 51 <= hr <= 100 else (1 if 101 <= hr <= 110 else 2)
            mews_sbp = 0 if 101 <= sbp <= 199 else (1 if 81 <= sbp <= 100 else 2)
            mews_total = mews_resp + mews_hr + mews_sbp
            
            abnormal_count = sum([
                temp > 100.4 or temp < 96,
                hr > 100 or hr < 60,
                rr > 20 or rr < 12,
                spo2 < 95,
                sbp > 140 or sbp < 90
            ])
            
            # Count diagnoses
            num_dx = sum([dx_mi, dx_stroke, dx_sepsis, dx_head, dx_resp, 
                         dx_heart_failure, dx_gi_bleed, dx_pe])
            
            # Count comorbidities
            num_comorbid = sum([pmh_cardiac, pmh_htn, pmh_respiratory, pmh_diabetes,
                               pmh_endocrine, pmh_renal, pmh_gu, pmh_blood,
                               pmh_infectious, pmh_msk, pmh_neuro, pmh_cancer, pmh_symptoms])
            
            # Build patient data
            patient_data = {
                'temperature': temp, 'heartrate': hr, 'resprate': rr,
                'o2sat': spo2, 'sbp': sbp, 'dbp': dbp, 'pain': pain,
                'age_at_visit': age, 'gender_male': 1 if gender == "Male" else 0,
                
                # Vital aggregates (count=5 for stable)
                'temperature_initial': temp, 'temperature_mean': temp,
                'temperature_min': temp, 'temperature_max': temp,
                'temperature_std': 0, 'temperature_trend': 0, 'temperature_count': 5,
                
                'heartrate_initial': hr, 'heartrate_mean': hr,
                'heartrate_min': hr, 'heartrate_max': hr,
                'heartrate_std': 0, 'heartrate_trend': 0, 'heartrate_count': 5,
                
                'resprate_initial': rr, 'resprate_mean': rr,
                'resprate_min': rr, 'resprate_max': rr,
                'resprate_std': 0, 'resprate_trend': 0, 'resprate_count': 5,
                
                'o2sat_initial': spo2, 'o2sat_mean': spo2,
                'o2sat_min': spo2, 'o2sat_max': spo2,
                'o2sat_std': 0, 'o2sat_trend': 0, 'o2sat_count': 5,
                
                'sbp_initial': sbp, 'sbp_mean': sbp,
                'sbp_min': sbp, 'sbp_max': sbp,
                'sbp_std': 0, 'sbp_trend': 0, 'sbp_count': 5,
                
                'dbp_initial': dbp, 'dbp_mean': dbp,
                'dbp_min': dbp, 'dbp_max': dbp,
                'dbp_std': 0, 'dbp_trend': 0, 'dbp_count': 5,
                
                # Derived
                'map_mean': map_val,
                'pulse_pressure_mean': sbp - dbp,
                'shock_index': shock_idx,
                
                # MEWS
                'mews_resp': mews_resp,
                'mews_hr': mews_hr,
                'mews_sbp': mews_sbp,
                'mews_total': mews_total,
                'abnormal_vitals_count': abnormal_count,
                
                # Chief complaints
                'cc_chest_pain': int(cc_chest),
                'cc_shortness_breath': int(cc_sob),
                'cc_abdominal_pain': int(cc_abd),
                'cc_neurological': int(cc_neuro),
                'cc_trauma': int(cc_trauma),
                'cc_infection': int(cc_infection),
                'cc_hemorrhage': int(cc_hemorrhage),
                'cc_cardiac': int(cc_chest),
                'cc_pain': int(pain >= 7),
                'cc_hypotension': int(sbp < 90),
                'cc_hypertension': int(sbp > 140),
                
                # Diagnoses
                'dx_mi': int(dx_mi),
                'dx_stroke': int(dx_stroke),
                'dx_sepsis': int(dx_sepsis),
                'dx_head_trauma': int(dx_head),
                'dx_resp_failure': int(dx_resp),
                'dx_heart_failure': int(dx_heart_failure),
                'dx_gi_bleed': int(dx_gi_bleed),
                'dx_pe': int(dx_pe),
                'num_diagnoses': num_dx,
            }
            
            # Predict
            acuity, reasoning, probs = predict_patient(patient_data, model, scaler, imputer, feature_cols)
            
            # Build display string for conditions
            conditions_list = []
            if dx_mi: conditions_list.append("MI")
            if dx_stroke: conditions_list.append("Stroke")
            if dx_sepsis: conditions_list.append("Sepsis")
            if dx_head: conditions_list.append("Head Trauma")
            if dx_resp: conditions_list.append("Resp Failure")
            if dx_heart_failure: conditions_list.append("CHF")
            if dx_gi_bleed: conditions_list.append("GI Bleed")
            if dx_pe: conditions_list.append("PE")
            
            comorbid_list = []
            if pmh_cardiac: comorbid_list.append("Cardiac")
            if pmh_htn: comorbid_list.append("HTN")
            if pmh_respiratory: comorbid_list.append("Resp")
            if pmh_diabetes: comorbid_list.append("DM")
            if pmh_renal: comorbid_list.append("Renal")
            if pmh_blood: comorbid_list.append("Blood")
            if pmh_cancer: comorbid_list.append("Cancer")
            
            complaints_list = []
            if cc_chest: complaints_list.append("Chest Pain")
            if cc_sob: complaints_list.append("SOB")
            if cc_abd: complaints_list.append("Abd Pain")
            if cc_neuro: complaints_list.append("Neuro")
            if cc_trauma: complaints_list.append("Trauma")
            if cc_infection: complaints_list.append("Infection")
            if cc_hemorrhage: complaints_list.append("Bleeding")
            
            # Add to queue
            patient = {
                'id': st.session_state.patient_counter,
                'name': name,
                'age': age,
                'gender': gender,
                'acuity': acuity,
                'reasoning': reasoning,
                'vitals': f"T:{temp}°F HR:{hr} RR:{rr} BP:{sbp}/{dbp} O2:{spo2}%",
                'pain': pain,
                'conditions': ', '.join(conditions_list) if conditions_list else 'None',
                'comorbidities': ', '.join(comorbid_list) if comorbid_list else 'None',
                'complaints': ', '.join(complaints_list) if complaints_list else 'None',
                'arrival_time': datetime.now(),
                'probabilities': probs,
                'data': patient_data,
                'num_dx': num_dx,
                'num_comorbid': num_comorbid
            }
            
            st.session_state.patient_queue.append(patient)
            st.session_state.patient_counter += 1
            st.success(f"✅ {name} added - Acuity Level {acuity}")
            st.rerun()

# Main
def main():
    model, scaler, imputer, feature_cols, metrics = load_model_artifacts()
    if not model:
        st.stop()
    
    # Sidebar
    render_sidebar(model, scaler, imputer, feature_cols)
    
    # Header
    st.title("🏥 Emergency Room Triage System")
    
    # Stats
    if st.session_state.patient_queue:
        col1, col2, col3, col4 = st.columns(4)
        queue = st.session_state.patient_queue
        
        col1.metric("Total Patients", len(queue))
        col2.metric("🚨 Critical (5)", len([p for p in queue if p['acuity'] == 5]))
        col3.metric("🔴 Very Urgent (4)", len([p for p in queue if p['acuity'] == 4]))
        col4.metric("⚠️ Urgent (3)", len([p for p in queue if p['acuity'] == 3]))
    
    st.markdown("---")
    
    # Queue
    if not st.session_state.patient_queue:
        st.info("👋 No patients in queue. Use the sidebar to add patients.")
    else:
        st.subheader("📋 Priority Queue (Sorted by Acuity)")
        
        sorted_queue = sorted(st.session_state.patient_queue,
                            key=lambda x: (-x['acuity'], x['arrival_time']))
        
        for patient in sorted_queue:
            acuity_labels = {5: "CRITICAL", 4: "VERY URGENT", 3: "URGENT", 
                           2: "SEMI-URGENT", 1: "NON-URGENT"}
            
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="acuity-card level-{patient['acuity']}">
                        <h1 style="margin:0;">Level {patient['acuity']}</h1>
                        <p style="margin:0; font-size:12px;">{acuity_labels[patient['acuity']]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.write(f"**{patient['name']}** - {patient['age']}yo {patient['gender']}")
                    st.write(f"🩺 {patient['vitals']}")
                    st.write(f"📋 CC: {patient['complaints']}")
                    st.write(f"🏥 Dx: {patient['conditions']}")
                    if patient.get('comorbidities', 'None') != 'None':
                        st.write(f"📊 PMH: {patient['comorbidities']}")
                
                with col3:
                    if st.button("📊", key=f"details_{patient['id']}"):
                        with st.expander("📈 Model Analysis", expanded=True):
                            st.write("**ML Confidence Distribution:**")
                            fig = px.bar(
                                x=[f"L{i}" for i in range(1, 6)],
                                y=patient['probabilities'],
                                labels={'x': 'Acuity', 'y': 'Probability'},
                                color=patient['probabilities'],
                                color_continuous_scale=['green', 'yellow', 'orange', 'red', 'purple']
                            )
                            fig.update_layout(showlegend=False, height=200)
                            st.plotly_chart(fig, use_container_width=True)
                            
                    
                    if st.button("❌", key=f"remove_{patient['id']}"):
                        st.session_state.patient_queue = [p for p in st.session_state.patient_queue 
                                                         if p['id'] != patient['id']]
                        st.rerun()
                
                st.markdown("---")
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.patient_queue = []
                st.rerun()
        with col2:
            df = pd.DataFrame([{
                'Name': p['name'], 'Age': p['age'], 'Acuity': p['acuity'],
                'Vitals': p['vitals'], 'Wait_Min': (datetime.now() - p['arrival_time']).seconds // 60
            } for p in sorted_queue])
            
            st.download_button("📥 Export CSV", df.to_csv(index=False),
                             f"er_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                             use_container_width=True)

if __name__ == "__main__":
    main()