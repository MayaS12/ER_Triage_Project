"""
Step 1: Data Preparation and Feature Engineering for ER Triage System
This script processes MIMIC-ED data to create a unified dataset for ML modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ICD-9 to ICD-10 mapping for common emergency conditions
ICD9_TO_ICD10_MAP = {
    '4589': 'I959',  # Hypotension
    '78650': 'R079',  # Chest pain
    '7862': 'R51',    # Headache
    '78900': 'R509',  # Fever
    '78959': 'R42',   # Dizziness
    '78609': 'R0689', # Dyspnea
    '78791': 'R104',  # Abdominal pain
    '7840': 'R110',   # Nausea/vomiting
    '78092': 'R5383', # Altered mental status
    '78552': 'R568',  # Shock
    '99591': 'R6521', # Sepsis
    '42731': 'I4891', # Atrial fibrillation
    '41401': 'I2101', # MI
    '51881': 'J189',  # Pneumonia
    '5789': 'K529',   # GI bleed
    '43491': 'I63',   # CVA/stroke
}


def load_data():
    """Load all CSV files"""
    print("Loading data files...")
    
    admissions = pd.read_csv('admissions.csv')
    diagnosis = pd.read_csv('diagnosis.csv')
    edstays = pd.read_csv('edstays.csv')
    patients = pd.read_csv('patients.csv')
    triage = pd.read_csv('triage.csv')
    vitalsign = pd.read_csv('vitalsign.csv')
    
    print(f"Loaded {len(admissions)} admissions")
    print(f"Loaded {len(diagnosis)} diagnoses")
    print(f"Loaded {len(edstays)} ED stays")
    print(f"Loaded {len(patients)} patients")
    print(f"Loaded {len(triage)} triage records")
    print(f"Loaded {len(vitalsign)} vital sign measurements")
    
    return admissions, diagnosis, edstays, patients, triage, vitalsign


def standardize_icd_codes(diagnosis_df):
    """Standardize ICD-9 codes to ICD-10"""
    print("\nStandardizing ICD codes...")
    
    diagnosis_df = diagnosis_df.copy()
    
    # Convert ICD-9 to ICD-10 where possible
    def convert_icd(row):
        if row['icd_version'] == 9:
            icd_code = str(row['icd_code'])
            if icd_code in ICD9_TO_ICD10_MAP:
                return ICD9_TO_ICD10_MAP[icd_code], 10
        return row['icd_code'], row['icd_version']
    
    diagnosis_df[['icd_code_std', 'icd_version_std']] = diagnosis_df.apply(
        lambda row: pd.Series(convert_icd(row)), axis=1
    )
    
    print(f"Converted {(diagnosis_df['icd_version'] == 9).sum()} ICD-9 codes")
    
    return diagnosis_df


def aggregate_vitals(vitalsign_df):
    """
    Aggregate multiple vital sign measurements per stay.
    Calculate initial, mean, min, max, and trend features.
    """
    print("\nAggregating vital signs...")
    
    # Convert charttime to datetime
    vitalsign_df['charttime'] = pd.to_datetime(vitalsign_df['charttime'])
    
    # Clean pain column - convert non-numeric values to NaN
    vitalsign_df['pain'] = pd.to_numeric(vitalsign_df['pain'], errors='coerce')
    
    # Sort by stay_id and charttime
    vitalsign_df = vitalsign_df.sort_values(['stay_id', 'charttime'])
    
    # Define vital sign columns
    vital_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
    
    aggregated = []
    
    for stay_id, group in vitalsign_df.groupby('stay_id'):
        features = {'stay_id': stay_id}
        
        for col in vital_cols:
            valid_values = group[col].dropna()
            
            if len(valid_values) > 0:
                # Initial (first recorded)
                features[f'{col}_initial'] = valid_values.iloc[0]
                
                # Statistical features
                features[f'{col}_mean'] = valid_values.mean()
                features[f'{col}_min'] = valid_values.min()
                features[f'{col}_max'] = valid_values.max()
                features[f'{col}_std'] = valid_values.std() if len(valid_values) > 1 else 0
                
                # Trend: difference between last and first
                if len(valid_values) > 1:
                    features[f'{col}_trend'] = valid_values.iloc[-1] - valid_values.iloc[0]
                else:
                    features[f'{col}_trend'] = 0
                
                # Count of measurements
                features[f'{col}_count'] = len(valid_values)
            else:
                # Missing data indicators
                features[f'{col}_initial'] = np.nan
                features[f'{col}_mean'] = np.nan
                features[f'{col}_min'] = np.nan
                features[f'{col}_max'] = np.nan
                features[f'{col}_std'] = 0
                features[f'{col}_trend'] = 0
                features[f'{col}_count'] = 0
        
        aggregated.append(features)
    
    vitals_agg = pd.DataFrame(aggregated)
    print(f"Aggregated vitals for {len(vitals_agg)} stays")
    
    return vitals_agg


def extract_chief_complaint_features(triage_df):
    """
    Extract features from chief complaint text.
    Create binary indicators for key symptoms and categories.
    """
    print("\nExtracting chief complaint features...")
    
    triage_df = triage_df.copy()
    
    # Convert to lowercase for matching
    triage_df['complaint_lower'] = triage_df['chiefcomplaint'].fillna('').str.lower()
    
    # Define symptom categories
    symptom_keywords = {
        'chest_pain': ['chest', 'cardiac', 'heart'],
        'shortness_breath': ['breath', 'dyspnea', 'sob', 'respiratory'],
        'abdominal_pain': ['abdom', 'stomach', 'belly'],
        'neurological': ['stroke', 'neuro', 'weakness', 'numbness', 'confusion', 'altered'],
        'trauma': ['trauma', 'fall', 'injury', 'accident', 'fracture'],
        'infection': ['fever', 'infection', 'sepsis'],
        'hemorrhage': ['bleed', 'blood', 'hemorrhage'],
        'cardiac': ['mi', 'infarct', 'arrest', 'arrhythmia', 'afib'],
        'pain': ['pain'],
        'hypotension': ['hypotension', 'low blood pressure'],
        'hypertension': ['hypertension', 'high blood pressure'],
    }
    
    # Create binary features
    for category, keywords in symptom_keywords.items():
        pattern = '|'.join(keywords)
        triage_df[f'cc_{category}'] = triage_df['complaint_lower'].str.contains(
            pattern, na=False, regex=True
        ).astype(int)
    
    # Drop temporary column
    triage_df = triage_df.drop('complaint_lower', axis=1)
    
    print(f"Created {len(symptom_keywords)} chief complaint features")
    
    return triage_df


def create_diagnosis_features(diagnosis_df):
    """
    Create diagnosis-based features.
    Count total diagnoses and create flags for high-risk conditions.
    """
    print("\nCreating diagnosis features...")
    
    # High-risk ICD-10 code prefixes
    high_risk_prefixes = {
        'I21': 'mi',           # Myocardial infarction
        'I63': 'stroke',       # Cerebral infarction
        'R65': 'sepsis',       # Sepsis
        'S06': 'head_trauma',  # Intracranial injury
        'I50': 'heart_failure', # Heart failure
        'J96': 'resp_failure',  # Respiratory failure
        'K92': 'gi_bleed',     # GI hemorrhage
        'I26': 'pe',           # Pulmonary embolism
    }
    
    diagnosis_features = []
    
    for stay_id, group in diagnosis_df.groupby('stay_id'):
        features = {
            'stay_id': stay_id,
            'num_diagnoses': len(group)
        }
        
        # Check for high-risk conditions
        for prefix, condition in high_risk_prefixes.items():
            features[f'dx_{condition}'] = int(
                group['icd_code_std'].astype(str).str.startswith(prefix).any()
            )
        
        diagnosis_features.append(features)
    
    dx_features_df = pd.DataFrame(diagnosis_features)
    print(f"Created diagnosis features for {len(dx_features_df)} stays")
    
    return dx_features_df


def calculate_age_at_visit(patients_df, edstays_df):
    """Calculate patient age at time of ED visit"""
    print("\nCalculating ages...")
    
    # Merge to get anchor info
    merged = edstays_df.merge(
        patients_df[['subject_id', 'anchor_age', 'anchor_year']], 
        on='subject_id'
    )
    
    # Extract year from intime
    merged['intime'] = pd.to_datetime(merged['intime'])
    merged['visit_year'] = merged['intime'].dt.year
    
    # Calculate age at visit
    merged['age_at_visit'] = merged['anchor_age'] + (merged['visit_year'] - merged['anchor_year'])
    
    return merged[['stay_id', 'age_at_visit']]


def create_derived_vital_features(df):
    """
    Create clinically relevant derived features from vitals.
    """
    print("\nCreating derived vital sign features...")
    
    # Mean Arterial Pressure (MAP)
    df['map_mean'] = ((2 * df['dbp_mean']) + df['sbp_mean']) / 3
    
    # Pulse Pressure (indicator of cardiovascular health)
    df['pulse_pressure_mean'] = df['sbp_mean'] - df['dbp_mean']
    
    # Shock Index (HR/SBP) - higher values indicate shock
    df['shock_index'] = df['heartrate_mean'] / df['sbp_mean']
    df['shock_index'] = df['shock_index'].replace([np.inf, -np.inf], np.nan)
    
    # Modified Early Warning Score (MEWS) components
    # Respiratory rate score
    df['mews_resp'] = 0
    df.loc[df['resprate_mean'] < 9, 'mews_resp'] = 2
    df.loc[(df['resprate_mean'] >= 9) & (df['resprate_mean'] <= 14), 'mews_resp'] = 0
    df.loc[(df['resprate_mean'] >= 15) & (df['resprate_mean'] <= 20), 'mews_resp'] = 1
    df.loc[(df['resprate_mean'] >= 21) & (df['resprate_mean'] <= 29), 'mews_resp'] = 2
    df.loc[df['resprate_mean'] >= 30, 'mews_resp'] = 3
    
    # Heart rate score
    df['mews_hr'] = 0
    df.loc[df['heartrate_mean'] < 40, 'mews_hr'] = 2
    df.loc[(df['heartrate_mean'] >= 40) & (df['heartrate_mean'] <= 50), 'mews_hr'] = 1
    df.loc[(df['heartrate_mean'] >= 51) & (df['heartrate_mean'] <= 100), 'mews_hr'] = 0
    df.loc[(df['heartrate_mean'] >= 101) & (df['heartrate_mean'] <= 110), 'mews_hr'] = 1
    df.loc[(df['heartrate_mean'] >= 111) & (df['heartrate_mean'] <= 129), 'mews_hr'] = 2
    df.loc[df['heartrate_mean'] >= 130, 'mews_hr'] = 3
    
    # Systolic BP score
    df['mews_sbp'] = 0
    df.loc[df['sbp_mean'] < 70, 'mews_sbp'] = 3
    df.loc[(df['sbp_mean'] >= 70) & (df['sbp_mean'] <= 80), 'mews_sbp'] = 2
    df.loc[(df['sbp_mean'] >= 81) & (df['sbp_mean'] <= 100), 'mews_sbp'] = 1
    df.loc[(df['sbp_mean'] >= 101) & (df['sbp_mean'] <= 199), 'mews_sbp'] = 0
    df.loc[df['sbp_mean'] >= 200, 'mews_sbp'] = 2
    
    # Total MEWS score
    df['mews_total'] = df['mews_resp'] + df['mews_hr'] + df['mews_sbp']
    
    # Abnormal vital signs count
    df['abnormal_vitals_count'] = 0
    df['abnormal_vitals_count'] += (df['temperature_mean'] > 38.0) | (df['temperature_mean'] < 36.0)
    df['abnormal_vitals_count'] += (df['heartrate_mean'] > 100) | (df['heartrate_mean'] < 60)
    df['abnormal_vitals_count'] += (df['resprate_mean'] > 20) | (df['resprate_mean'] < 12)
    df['abnormal_vitals_count'] += (df['o2sat_mean'] < 95)
    df['abnormal_vitals_count'] += (df['sbp_mean'] > 140) | (df['sbp_mean'] < 90)
    
    print(f"Created {8} derived vital sign features")
    
    return df


def merge_all_data(admissions, diagnosis, edstays, patients, triage, vitalsign):
    """
    Main function to merge all data sources and create final dataset.
    """
    print("\n" + "="*80)
    print("STARTING DATA MERGE AND FEATURE ENGINEERING")
    print("="*80)
    
    # 1. Standardize ICD codes
    diagnosis = standardize_icd_codes(diagnosis)
    
    # 2. Aggregate vital signs
    vitals_agg = aggregate_vitals(vitalsign)
    
    # 3. Extract chief complaint features
    triage = extract_chief_complaint_features(triage)
    
    # 4. Create diagnosis features
    dx_features = create_diagnosis_features(diagnosis)
    
    # 5. Calculate ages
    age_df = calculate_age_at_visit(patients, edstays)
    
    # 6. Start merging - use triage as base since it has acuity (target variable)
    print("\nMerging datasets...")
    
    # Start with triage (has acuity)
    merged = triage.copy()
    
    # Clean pain column - convert non-numeric values to NaN
    merged['pain'] = pd.to_numeric(merged['pain'], errors='coerce')
    
    print(f"Base: {len(merged)} triage records")
    
    # Add ED stays info
    merged = merged.merge(
        edstays[['stay_id', 'subject_id', 'hadm_id', 'gender', 'race', 'arrival_transport', 'disposition']],
        on=['subject_id', 'stay_id'],
        how='left'
    )
    print(f"After edstays: {len(merged)} records")
    
    # Add age
    merged = merged.merge(age_df, on='stay_id', how='left')
    print(f"After age: {len(merged)} records")
    
    # Add aggregated vitals
    merged = merged.merge(vitals_agg, on='stay_id', how='left')
    print(f"After vitals: {len(merged)} records")
    
    # Add diagnosis features
    merged = merged.merge(dx_features, on='stay_id', how='left')
    print(f"After diagnosis: {len(merged)} records")
    
    # Fill missing diagnosis features with 0
    dx_cols = [col for col in merged.columns if col.startswith('dx_')]
    merged[dx_cols] = merged[dx_cols].fillna(0)
    merged['num_diagnoses'] = merged['num_diagnoses'].fillna(0)
    
    # 7. Create derived features
    merged = create_derived_vital_features(merged)
    
    # 8. Encode categorical variables
    print("\nEncoding categorical variables...")
    
    # Gender
    merged['gender_male'] = (merged['gender'] == 'M').astype(int)
    
    # Race - create binary indicators for major groups
    race_dummies = pd.get_dummies(merged['race'], prefix='race')
    merged = pd.concat([merged, race_dummies], axis=1)
    
    # Note: Transport features are NOT encoded to avoid bias
    # arrival_transport will be excluded from modeling
    
    # 9. Create time-based features if needed
    # Note: This could be expanded with day of week, time of day, etc.
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\nFinal dataset shape: {merged.shape}")
    print(f"Total features: {merged.shape[1]}")
    print(f"\nAcuity distribution:")
    print(merged['acuity'].value_counts().sort_index())
    print(f"\nMissing values by column:")
    missing = merged.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))
    
    return merged


def save_processed_data(df, filename='processed_ed_data.csv'):
    """Save the processed dataset"""
    df.to_csv(filename, index=False)
    print(f"\n✓ Saved processed data to {filename}")
    
    # Also save a summary
    summary = {
        'total_records': len(df),
        'total_features': df.shape[1],
        'acuity_distribution': df['acuity'].value_counts().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    import json
    with open('data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved data summary to data_summary.json")


if __name__ == "__main__":
    # Load data
    admissions, diagnosis, edstays, patients, triage, vitalsign = load_data()
    
    # Process and merge
    processed_data = merge_all_data(admissions, diagnosis, edstays, patients, triage, vitalsign)
    
    # Save
    save_processed_data(processed_data)
    
    print("\n" + "="*80)
    print("STEP 1 COMPLETE - Ready for modeling!")
    print("="*80)