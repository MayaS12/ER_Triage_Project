"""Test what features are causing the issue"""
import pandas as pd
import numpy as np

# Load feature columns from your actual file
with open('feature_columns.txt', 'r') as f:
    feature_cols = [line.strip() for line in f.readlines()]

print(f"Model expects {len(feature_cols)} features:")
print("\n".join(feature_cols))

# Check which features are NOT in our default dict
dashboard_features = [
    'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain',
    'cc_chest_pain', 'cc_shortness_breath', 'cc_abdominal_pain', 
    'cc_neurological', 'cc_trauma', 'cc_infection', 'cc_hemorrhage',
    'cc_cardiac', 'cc_pain', 'cc_hypotension', 'cc_hypertension',
    'age_at_visit', 'num_diagnoses',
    'dx_mi', 'dx_stroke', 'dx_sepsis', 'dx_head_trauma',
    'dx_heart_failure', 'dx_resp_failure', 'dx_gi_bleed', 'dx_pe',
    'gender_male',
    # Vital aggregates
    'temperature_initial', 'temperature_mean', 'temperature_min', 'temperature_max',
    'temperature_std', 'temperature_trend', 'temperature_count',
    'heartrate_initial', 'heartrate_mean', 'heartrate_min', 'heartrate_max',
    'heartrate_std', 'heartrate_trend', 'heartrate_count',
    'resprate_initial', 'resprate_mean', 'resprate_min', 'resprate_max',
    'resprate_std', 'resprate_trend', 'resprate_count',
    'o2sat_initial', 'o2sat_mean', 'o2sat_min', 'o2sat_max',
    'o2sat_std', 'o2sat_trend', 'o2sat_count',
    'sbp_initial', 'sbp_mean', 'sbp_min', 'sbp_max',
    'sbp_std', 'sbp_trend', 'sbp_count',
    'dbp_initial', 'dbp_mean', 'dbp_min', 'dbp_max',
    'dbp_std', 'dbp_trend', 'dbp_count',
    # Derived
    'map_mean', 'pulse_pressure_mean', 'shock_index',
    'mews_total', 'mews_resp', 'mews_hr', 'mews_sbp',
    'abnormal_vitals_count',
]

print(f"\n\nDashboard has {len(dashboard_features)} features")

missing = set(feature_cols) - set(dashboard_features)
extra = set(dashboard_features) - set(feature_cols)

if missing:
    print(f"\n❌ MISSING from dashboard ({len(missing)}):")
    for f in sorted(missing):
        print(f"  - {f}")

if extra:
    print(f"\n⚠️ EXTRA in dashboard (not used by model) ({len(extra)}):")
    for f in sorted(extra):
        print(f"  - {f}")