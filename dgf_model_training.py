import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
import os

def build_vanilla_dgf_model(df):
    """
    Build a simple DGF prediction model using only donor variables
    """
    
    print("=== VANILLA DGF MODEL - DONOR VARIABLES ONLY ===\n")
    
    # Define core donor features
    donor_features = [
        'Donor_age',
        'Donor_final_creatinine',
        'Donor_eGFR_CKD_EPI_final_creatinine',
        'Cold_Ischemia_hours',
        'Expanded_criteria_donor',
        'KDRI_2024',
        # 'KDPI_2024'  # Often correlated with KDRI, start with just KDRI
    ]
    
    target = 'Delayed_Graft_Function'
    
    # Check which features are available in the dataset
    available_features = [f for f in donor_features if f in df.columns]
    missing_features = [f for f in donor_features if f not in df.columns]
    
    print(f"Available donor features ({len(available_features)}):")
    for f in available_features:
        print(f"   • {f}")
    
    if missing_features:
        print(f"\nMissing features ({len(missing_features)}):")
        for f in missing_features:
            print(f"   • {f}")
    
    # Check if target exists
    if target not in df.columns:
        print(f"\nERROR: Target variable '{target}' not found in dataset!")
        return None
    
    # Create modeling dataset
    model_data = df[available_features + [target]].copy()
    
    print(f"\nDataset Overview:")
    print(f"   • Total rows: {len(model_data)}")
    print(f"   • DGF cases: {model_data[target].sum()} ({model_data[target].mean():.1%})")
    print(f"   • No DGF cases: {(model_data[target] == 0).sum()} ({(model_data[target] == 0).mean():.1%})")
    
    # Handle missing values
    print(f"\nMissing Values Check:")
    missing_counts = model_data.isnull().sum()
    for feature in available_features + [target]:
        missing = missing_counts[feature]
        if missing > 0:
            print(f"   • {feature}: {missing} missing ({missing/len(model_data):.1%})")
        else:
            print(f"   • {feature}: No missing values")
    
    # Simple approach: drop rows with any missing values
    original_size = len(model_data)
    model_data = model_data.dropna()
    final_size = len(model_data)
    
    if final_size < original_size:
        print(f"\nDropped {original_size - final_size} rows with missing values")
        print(f"   Final dataset: {final_size} rows")
    
    if len(model_data) == 0:
        print("\nERROR: No complete cases remaining after dropping missing values!")
        return None
    
    # Feature summary statistics
    print(f"\nFeature Summary:")
    print(model_data[available_features].describe())
    
    return model_data, available_features

def train_and_evaluate_model(model_data, features, target='Delayed_Graft_Function'):
    """
    Train the logistic regression model and evaluate performance
    """
    
    print(f"\n=== MODEL TRAINING ===")
    
    # Prepare features and target
    X = model_data[features]
    y = model_data[target]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train vanilla logistic regression
    model = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        class_weight='balanced'  # Handle class imbalance if needed
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate performance
    print(f"\n=== MODEL PERFORMANCE ===")
    
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    train_acc = (y_train_pred == y_train).mean()
    test_acc = (y_test_pred == y_test).mean()
    
    print(f"TRAINING Performance:")
    print(f"   • AUC-ROC: {train_auc:.3f}")
    print(f"   • Accuracy: {train_acc:.3f}")
    
    print(f"\nTEST Performance:")
    print(f"   • AUC-ROC: {test_auc:.3f}")
    print(f"   • Accuracy: {test_acc:.3f}")
    
    # Check for overfitting
    auc_diff = train_auc - test_auc
    if auc_diff > 0.05:
        print(f"\nPotential overfitting detected (AUC difference: {auc_diff:.3f})")
    else:
        print(f"\nModel generalizes well (AUC difference: {auc_diff:.3f})")
    
    # Detailed test set evaluation
    print(f"\nDetailed Test Set Results:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    print("\nConfusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    
    # Clinical interpretation
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nClinical Metrics:")
    print(f"   • Sensitivity (True Positive Rate): {sensitivity:.3f}")
    print(f"   • Specificity (True Negative Rate): {specificity:.3f}")
    print(f"   • Positive Predictive Value: {ppv:.3f}")
    print(f"   • Negative Predictive Value: {npv:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0]),
        'Impact': ['Increases DGF Risk' if c > 0 else 'Decreases DGF Risk' for c in model.coef_[0]]
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"\nFeature Importance (Most to Least Important):")
    for _, row in feature_importance.iterrows():
        coef_str = f"{row['Coefficient']:+.3f}"
        print(f"   • {row['Feature']}: {coef_str} ({row['Impact']})")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    
    plt.figure(figsize=(10, 4))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, linewidth=2, label=f'Model (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - DGF Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature Importance
    plt.subplot(1, 2, 2)
    feature_importance_sorted = feature_importance.sort_values('Coefficient')
    colors = ['red' if x > 0 else 'blue' for x in feature_importance_sorted['Coefficient']]
    plt.barh(range(len(features)), feature_importance_sorted['Coefficient'], color=colors, alpha=0.7)
    plt.yticks(range(len(features)), feature_importance_sorted['Feature'])
    plt.xlabel('Coefficient (Log Odds)')
    plt.title('Feature Impact on DGF Risk')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Clinical interpretation of AUC
    print(f"\nClinical Utility Assessment:")
    if test_auc >= 0.80:
        print("   EXCELLENT: Model has strong clinical utility")
    elif test_auc >= 0.75:
        print("   GOOD: Model has good clinical utility") 
    elif test_auc >= 0.70:
        print("   FAIR: Model has modest clinical utility")
    elif test_auc >= 0.60:
        print("   POOR: Model has limited clinical utility")
    else:
        print("   INADEQUATE: Model not suitable for clinical use")
    
    return model, scaler, feature_importance

def save_model(model, scaler, features, feature_importance, prefix='vanilla_dgf'):
    """
    Save the trained model components
    """
    print(f"\n=== SAVING MODEL ===")
    
    # Save model components
    joblib.dump(model, f'{prefix}_model.pkl')
    joblib.dump(scaler, f'{prefix}_scaler.pkl')
    joblib.dump(features, f'{prefix}_features.pkl')
    
    # Save feature importance for reference
    feature_importance.to_csv(f'{prefix}_feature_importance.csv', index=False)
    
    print(f"Model saved successfully:")
    print(f"   • {prefix}_model.pkl")
    print(f"   • {prefix}_scaler.pkl") 
    print(f"   • {prefix}_features.pkl")
    print(f"   • {prefix}_feature_importance.csv")
    
    return True

# Main execution function
def run_vanilla_dgf_pipeline(data_file):
    """
    Complete pipeline: load data → build model → evaluate → save
    """
    print("Starting Vanilla DGF Model Pipeline...\n")
    
    # Load data
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Build model dataset
    result = build_vanilla_dgf_model(df)
    if result is None:
        print("Failed to build model dataset")
        return None
    
    model_data, features = result
    
    # Train and evaluate
    model, scaler, feature_importance = train_and_evaluate_model(model_data, features)
    
    # Save model
    save_model(model, scaler, features, feature_importance)
    
    print(f"\nPipeline completed successfully!")
    print(f"Ready to integrate with Streamlit app")
    
    return model, scaler, features, feature_importance

# Example usage
if __name__ == "__main__":
    # Replace with your actual data file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "Reduced_Features_NO_COLOR.csv")
    
    
    print("To run the pipeline:")
    print(f"result = run_vanilla_dgf_pipeline('{data_file}')")
    print("\nOr step by step:")
    print("df = pd.read_csv('Reduced_Features_NO_COLOR.csv')")
    print("model_data, features = build_vanilla_dgf_model(df)")
    print("model, scaler, importance = train_and_evaluate_model(model_data, features)")
    print("save_model(model, scaler, features, importance)")


result = run_vanilla_dgf_pipeline(data_file)