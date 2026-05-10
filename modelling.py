"""
Step 2: ML Modeling for ER Triage System
Predicts acuity levels (1-5) with special attention to critical cases (level 5).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, matthews_corrcoef
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(filename='processed_ed_data.csv'):
    """Load the processed dataset from step 1"""
    print("Loading processed data...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} records with {df.shape[1]} features")
    print(f"\nAcuity distribution:")
    print(df['acuity'].value_counts().sort_index())
    return df


def prepare_features(df):
    """
    Prepare features for modeling.
    Separate features from target and handle missing values.
    """
    print("\nPreparing features for modeling...")
    
    # Define target
    target = 'acuity'
    
    # Clean acuity values - ensure they are 1-5 only
    print("Cleaning acuity values...")
    df = df.copy()
    df['acuity'] = pd.to_numeric(df['acuity'], errors='coerce')
    
    # Remove any rows where acuity is not between 1 and 5
    valid_mask = (df['acuity'] >= 1) & (df['acuity'] <= 5)
    print(f"Removing {(~valid_mask).sum()} records with invalid acuity values")
    df = df[valid_mask]
    
    # Define columns to exclude from features
    exclude_cols = [
        'subject_id', 'stay_id', 'hadm_id', 'acuity',
        'chiefcomplaint',  # text field, already encoded
        'gender', 'race', 'arrival_transport', 'disposition',  # already encoded
        'intime', 'outtime', 'charttime'  # datetime fields
    ]
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # IMPORTANT: Remove all race-related features to prevent bias
    race_cols = [col for col in feature_cols if col.startswith('race_')]
    if race_cols:
        print(f"Removing {len(race_cols)} race-related features to prevent bias")
        feature_cols = [col for col in feature_cols if not col.startswith('race_')]
    
    print(f"Using {len(feature_cols)} features (excluding race)")
    
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    # Convert object/string columns to numeric where possible
    print("Converting data types...")
    for col in X.columns:
        if X[col].dtype == 'object':
            # Try to convert to numeric, coercing errors to NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    print("Handling missing values...")
    
    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Impute numeric columns with median
    numeric_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
    
    # Impute categorical columns with most frequent
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
    
    print(f"✓ Missing values handled")
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    print(f"Final dataset: {len(X)} samples")
    print(f"Acuity distribution in final dataset:")
    print(y.value_counts().sort_index())
    
    return X, y, feature_cols, numeric_imputer, categorical_imputer if len(categorical_cols) > 0 else None


def create_custom_weights(y):
    """
    Create custom class weights with strong emphasis on acuity level 5.
    Uses exponential scaling to heavily penalize misclassification of critical cases.
    """
    print("\nCalculating custom class weights...")
    
    # Get unique classes present in y
    unique_classes = np.unique(y.astype(int))
    n_samples = len(y)
    
    # Count samples per class
    class_counts = {}
    for cls in unique_classes:
        class_counts[cls] = np.sum(y == cls)
    
    # Calculate base weights (inverse frequency)
    custom_weights = {}
    for cls in unique_classes:
        count = class_counts[cls]
        base_weight = n_samples / (len(unique_classes) * count)
        
        if cls == 5:
            # Critical - multiply base weight by 15 
            custom_weights[cls] = base_weight * 15
        elif cls == 4:
            # Very Urgent - multiply base weight by 8
            custom_weights[cls] = base_weight * 8
        elif cls == 3:
            # Urgent - multiply base weight by 2
            custom_weights[cls] = base_weight * 2
        elif cls == 2:
            # Semi-urgent - multiply base weight by 1.3
            custom_weights[cls] = base_weight * 1.3
        else:
            # Non-urgent - use base weight
            custom_weights[cls] = base_weight
    
    print("Class weights (with critical emphasis):")
    for level in sorted(custom_weights.keys()):
        weight = custom_weights[level]
        count = class_counts[level]
        print(f"  Acuity {level}: {weight:.2f} (n={count})")
    
    return custom_weights


def create_sample_weights(y, custom_weights):
    """Create sample weights array from class weights"""
    sample_weights = np.array([custom_weights[int(label)] for label in y])
    return sample_weights


def train_xgboost_model(X_train, y_train, X_val, y_val, custom_weights):
    """
    Train XGBoost classifier with custom loss and parameters optimized for
    imbalanced medical triage data.
    """
    print("\nTraining XGBoost model...")
    
    # Convert acuity levels to 0-indexed for XGBoost
    y_train_indexed = y_train - 1
    y_val_indexed = y_val - 1
    n_classes = len(np.unique(y_train))
    
    # Create sample weights
    sample_weights = create_sample_weights(y_train, custom_weights)
    
    # XGBoost parameters - balanced for both recall and precision
    params = {
        'objective': 'multi:softprob',  # Changed to softprob for probability outputs
        'num_class': n_classes,
        'max_depth': 7,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 80,
        'tree_method': 'hist',
    }
    
    model = xgb.XGBClassifier(**params)
    
    # Train with sample weights
    model.fit(
        X_train, y_train_indexed,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_indexed)],
        verbose=False
    )
    
    print(f"✓ Model trained (best iteration: {model.best_iteration})")
    
    return model


def evaluate_model(model, X_test, y_test, custom_weights):
    """
    Comprehensive model evaluation with focus on critical case detection.
    Uses custom threshold for critical cases.
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test)
    
    # Standard predictions (convert back from 0-indexed)
    y_pred_indexed = model.predict(X_test)
    y_pred = y_pred_indexed + 1
    
    # Custom threshold adjustment for critical cases
    # Sweet spot threshold to balance recall and precision
    critical_threshold = 0.05  # 5% probability threshold
    urgent_threshold = 0.30    # 30% for acuity 4
    y_pred_adjusted = y_pred.copy()
    
    # Identify cases with high probability of being critical
    for i in range(len(y_pred_proba)):
        prob_acuity_5 = y_pred_proba[i][4]  # Index 4 = acuity 5 (0-indexed)
        prob_acuity_4 = y_pred_proba[i][3]  # Index 3 = acuity 4
        
        # Flag as critical if moderate signal
        if prob_acuity_5 > critical_threshold:
            y_pred_adjusted[i] = 5
        elif prob_acuity_4 > urgent_threshold and y_pred_adjusted[i] < 4:
            y_pred_adjusted[i] = 4
    
    print("\n[Using threshold-adjusted predictions for critical cases]")
    
    # Overall metrics with adjusted predictions
    print("\nOVERALL METRICS (Threshold-Adjusted):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_adjusted):.4f}")
    print(f"Weighted F1-Score: {f1_score(y_test, y_pred_adjusted, average='weighted'):.4f}")
    print(f"Macro F1-Score: {f1_score(y_test, y_pred_adjusted, average='macro'):.4f}")
    
    # Per-class metrics with threshold adjustment
    print("\nPER-CLASS METRICS (Threshold-Adjusted):")
    print(classification_report(y_test, y_pred_adjusted, 
                                target_names=[f'Acuity {i}' for i in range(1, 6)],
                                digits=4))
    
    # Critical case detection (Acuity 5)
    print("\n" + "="*80)
    print("CRITICAL CASE DETECTION (ACUITY 5) - THRESHOLD-ADJUSTED")
    print("="*80)
    
    # Binary classification: Acuity 5 vs. others
    y_test_binary = (y_test == 5).astype(int)
    y_pred_binary = (y_pred_adjusted == 5).astype(int)
    
    critical_recall = recall_score(y_test_binary, y_pred_binary)
    critical_precision = precision_score(y_test_binary, y_pred_binary)
    critical_f1 = f1_score(y_test_binary, y_pred_binary)
    
    print(f"Recall (Sensitivity): {critical_recall:.4f}")
    print(f"  → {critical_recall*100:.1f}% of critical cases correctly identified")
    print(f"Precision: {critical_precision:.4f}")
    print(f"  → {critical_precision*100:.1f}% of predicted critical cases are truly critical")
    print(f"F1-Score: {critical_f1:.4f}")
    
    # False negatives analysis (missed critical cases)
    false_negatives = np.sum((y_test == 5) & (y_pred_adjusted != 5))
    total_critical = np.sum(y_test == 5)
    print(f"\nMissed critical cases: {false_negatives} out of {total_critical}")
    
    if false_negatives > 0:
        missed_indices = np.where((y_test == 5) & (y_pred_adjusted != 5))[0]
        missed_predictions = y_pred_adjusted[missed_indices]
        print(f"Misclassified as: {np.bincount(missed_predictions.astype(int))}")
    
    # High-risk cases (Acuity 4 or 5)
    print("\n" + "="*80)
    print("HIGH-RISK CASE DETECTION (ACUITY 4-5) - THRESHOLD-ADJUSTED")
    print("="*80)
    
    y_test_highrisk = (y_test >= 4).astype(int)
    y_pred_highrisk = (y_pred_adjusted >= 4).astype(int)
    
    highrisk_recall = recall_score(y_test_highrisk, y_pred_highrisk)
    highrisk_precision = precision_score(y_test_highrisk, y_pred_highrisk)
    
    print(f"Recall: {highrisk_recall:.4f}")
    print(f"Precision: {highrisk_precision:.4f}")
    
    # Confusion matrix
    print("\nCONFUSION MATRIX (Threshold-Adjusted):")
    cm = confusion_matrix(y_test, y_pred_adjusted)
    print(cm)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, "confusion_matrix.png")
    
    # Also show standard predictions for comparison
    print("\n" + "="*80)
    print("STANDARD PREDICTIONS (No Threshold Adjustment) - For Comparison")
    print("="*80)
    y_test_binary_std = (y_test == 5).astype(int)
    y_pred_binary_std = (y_pred == 5).astype(int)
    print(f"Critical Recall (Standard): {recall_score(y_test_binary_std, y_pred_binary_std):.4f}")
    print(f"Critical Precision (Standard): {precision_score(y_test_binary_std, y_pred_binary_std):.4f}")
    
    return {
        'accuracy': accuracy_score(y_test, y_pred_adjusted),
        'f1_weighted': f1_score(y_test, y_pred_adjusted, average='weighted'),
        'critical_recall': critical_recall,
        'critical_precision': critical_precision,
        'critical_f1': critical_f1,
        'highrisk_recall': highrisk_recall,
        'highrisk_precision': highrisk_precision,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, filename):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Pred {i}' for i in range(1, 6)],
                yticklabels=[f'True {i}' for i in range(1, 6)])
    plt.title('Confusion Matrix - ER Triage Acuity Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('True Acuity', fontsize=12)
    plt.xlabel('Predicted Acuity', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {filename}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot top N most important features"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(top_n), y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to feature_importance.png")
    plt.close()
    
    # Save to CSV
    importance_df.to_csv('feature_importance.csv', index=False)
    print(f"✓ Feature importance saved to feature_importance.csv")
    
    return importance_df


def save_model_artifacts(model, scaler, imputer, feature_cols, metrics):
    """Save model and preprocessing artifacts"""
    print("\nSaving model artifacts...")
    
    # Save model
    joblib.dump(model, 'triage_model.pkl')
    print("✓ Model saved to triage_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Scaler saved to scaler.pkl")
    
    # Save imputer
    joblib.dump(imputer, 'imputer.pkl')
    print("✓ Imputer saved to imputer.pkl")
    
    # Save feature columns
    with open('feature_columns.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print("✓ Feature columns saved to feature_columns.txt")
    
    # Save metrics
    import json
    with open('model_metrics.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_serializable = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                               for k, v in metrics.items() if k != 'confusion_matrix'}
        json.dump(metrics_serializable, f, indent=2)
    print("✓ Metrics saved to model_metrics.json")


def main():
    print("="*80)
    print("ER TRIAGE ML MODEL TRAINING")
    print("="*80)
    
    # 1. Load processed data
    df = load_processed_data()
    
    # 2. Prepare features
    X, y, feature_cols, numeric_imputer, categorical_imputer = prepare_features(df)
    
    # 3. Split data (stratified to maintain class distribution)
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # 4. Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    # 5. Create custom weights
    custom_weights = create_custom_weights(y_train)
    
    # 6. Train model
    model = train_xgboost_model(
        X_train_scaled, y_train, 
        X_val_scaled, y_val, 
        custom_weights
    )
    
    # 7. Evaluate model
    metrics = evaluate_model(model, X_test_scaled, y_test, custom_weights)
    
    # 8. Plot feature importance
    plot_feature_importance(model, feature_cols)
    
    # 9. Save everything
    save_model_artifacts(model, scaler, numeric_imputer, feature_cols, metrics)
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE")
    print("="*80)
    print("\nKey Metrics:")
    print(f"  Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Critical Case Recall: {metrics['critical_recall']:.4f}")
    print(f"  Critical Case Precision: {metrics['critical_precision']:.4f}")
    print(f"  High-Risk Recall: {metrics['highrisk_recall']:.4f}")
    print("\nFiles generated:")
    print("  - triage_model.pkl")
    print("  - scaler.pkl")
    print("  - imputer.pkl")
    print("  - feature_columns.txt")
    print("  - model_metrics.json")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - feature_importance.csv")
    print("\n✓ Ready for deployment!")


if __name__ == "__main__":
    main()