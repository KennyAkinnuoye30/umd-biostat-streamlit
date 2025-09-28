"""  # --- Required libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import zipfile

# --- Load model ---
@st.cache_resource
def load_model():
    try:
        from input_clust_CAMEL import PathomicEnsemble
        model_path = os.path.join(os.path.dirname(__file__), 'pathomic_ensemble_model_fixed.pkl')
        return PathomicEnsemble.load_model(model_path)

    except Exception as e:
        st.error(f"❌ Model load failed: {e}")  # This will show the real issue in the UI

        # Fake model for testing
        class FakeModel:
            def predict(self, X_list, clinical_df, verbose=False):
                return np.random.uniform(10, 100, len(X_list))
        st.warning("⚠️ No model file found. Using a fake model for UI testing.")
        return FakeModel()

model = load_model()

# --- UI ---
st.title("Renal Outcome Predictor - Camelomics")
st.markdown("**Step 1:** Upload ZIP → **Step 2:** Enter clinical features → **Step 3:** Get predictions")

# Upload ZIP
pathomic_zip = st.file_uploader("Upload ZIP of Pathomic Excel Files", type="zip")

if pathomic_zip:
    # Process ZIP and show number of subjects
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "pathomics.zip")
        with open(zip_path, "wb") as f:
            f.write(pathomic_zip.getbuffer())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        excel_files = [f for f in os.listdir(tmp_dir) if f.endswith(".xlsx")]
        n_subjects = len(excel_files)
        
        st.success(f"Found {n_subjects} subjects")
        
        # Clinical data entry
        st.subheader("Enter Clinical Features")
        
        if n_subjects == 1:
            # Single entry
            age = st.number_input("Age", min_value=0, max_value=120, value=55)
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=26.0)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            creatinine = st.number_input("Creatinine", min_value=0.0, max_value=10.0, value=1.0)
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            clinical_data = [{
                'age': age, 'bmi': bmi, 'gender': gender, 
                'creatinine': creatinine, 'hypertension': hypertension
            }] * n_subjects
            
        else:
            # Multiple entry
            clinical_data = []
            for i in range(n_subjects):
                st.write(f"**Subject {i+1}**")
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age", value=55, key=f"age_{i}")
                    bmi = st.number_input("BMI", value=26.0, key=f"bmi_{i}")
                with col2:
                    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key=f"gender_{i}")
                    creatinine = st.number_input("Creatinine", value=1.0, key=f"creatinine_{i}")
                    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=f"hypertension_{i}")
                
                clinical_data.append({
                    'age': age, 'bmi': bmi, 'gender': gender,
                    'creatinine': creatinine, 'hypertension': hypertension
                })
        
        clinical_df = pd.DataFrame(clinical_data)
        
        # Run prediction
        if st.button("Generate Predictions"):
            # Process Excel files
            X_list = []
            for fname in sorted(excel_files):
                df = pd.read_excel(os.path.join(tmp_dir, fname))
                if "compartment_id" in df.columns:
                    df = df.drop(columns=["compartment_id"])
                X_list.append(df.to_numpy())
            
            # Make predictions
            predictions = model.predict(X_list, clinical_df, verbose=False)
            
            # Show results
            results_df = pd.DataFrame({
                "Subject": [f"Subject_{i+1}" for i in range(len(predictions))],
                "Filename": excel_files,
                "Predicted_Outcome": np.round(predictions, 2)
            })
            
            st.subheader("Results")
            st.dataframe(results_df)
            
            # Download
            st.download_button(
                "Download CSV",
                results_df.to_csv(index=False),
                "predictions.csv"
            )

else:
    st.info("Upload a ZIP file to continue.")
    
    # Show preview of what clinical features will be needed
    st.subheader("Clinical Features You'll Enter (After ZIP Upload)")
    st.write("Once you upload the ZIP file, you'll be prompted to enter:")
    st.write("• **Age** (years)")
    st.write("• **BMI** (body mass index)")  
    st.write("• **Gender** (Male/Female)")
    st.write("• **Creatinine** (mg/dL)")
    st.write("• **Hypertension** (Yes/No)")

"""

# --- Required libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Load DGF model ---
@st.cache_resource
def load_dgf_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'vanilla_dgf_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'vanilla_dgf_scaler.pkl')
        features_path = os.path.join(os.path.dirname(__file__), 'vanilla_dgf_features.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        
        return model, scaler, features
    
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        
        # Fake model for testing
        class FakeModel:
            def predict(self, X):
                return np.random.choice([0, 1], size=len(X))
            def predict_proba(self, X):
                return np.random.uniform(0, 1, size=(len(X), 2))
        
        class FakeScaler:
            def transform(self, X):
                return X
        
        fake_features = ['Donor_age', 'Cold_Ischemia_hours', 'Recipient_age_at_transplantation_date']
        
        st.warning("⚠️ No model file found. Using fake model for UI testing.")
        return FakeModel(), FakeScaler(), fake_features

model, scaler, model_features = load_dgf_model()

# --- UI ---
st.title("🏥 Delayed Graft Function (DGF) Predictor")
st.markdown("**Predict the risk of delayed graft function after kidney transplantation**")

st.markdown("---")

# Input mode selection
input_mode = st.radio(
    "Choose input method:",
    ["Single Patient", "Multiple Patients (CSV Upload)", "Manual Entry for Multiple Patients"]
)

def get_feature_input(prefix="", key_suffix=""):
    """Create input widgets for DGF model features - DONOR ONLY"""
    inputs = {}
    
    st.subheader(f"🫘 Donor Information {prefix}")
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['Donor_age'] = st.number_input(
            "Donor Age (years)", 
            min_value=0, max_value=85, value=45, 
            key=f"donor_age{key_suffix}"
        )
        inputs['Donor_final_creatinine'] = st.number_input(
            "Donor Final Creatinine (mg/dL)", 
            min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            key=f"donor_creat{key_suffix}"
        )
        inputs['Cold_Ischemia_hours'] = st.number_input(
            "Cold Ischemia Time (hours)", 
            min_value=0.0, max_value=48.0, value=12.0, step=0.5,
            key=f"cold_isch{key_suffix}"
        )
    
    with col2:
        inputs['Expanded_criteria_donor'] = st.selectbox(
            "Expanded Criteria Donor", 
            [0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes",
            key=f"ecd{key_suffix}"
        )
        inputs['KDRI_2024'] = st.number_input(
            "KDRI Score", 
            min_value=0.5, max_value=3.0, value=1.2, step=0.1,
            key=f"kdri{key_suffix}"
        )
        inputs['Donor_eGFR_CKD_EPI_final_creatinine'] = st.number_input(
            "Donor eGFR (mL/min/1.73m²)", 
            min_value=10, max_value=150, value=80,
            key=f"donor_egfr{key_suffix}"
        )
    
    return inputs

def prepare_model_input(inputs_dict, model_features):
    """Convert UI inputs to model-ready format"""
    # Only use features that the model was trained on
    available_inputs = {k: v for k, v in inputs_dict.items() if k in model_features}
    
    # Create feature array in correct order
    feature_values = []
    for feature in model_features:
        if feature in available_inputs:
            feature_values.append(available_inputs[feature])
        else:
            # Use default values for missing features
            feature_values.append(0)  # You might want more sophisticated defaults
    
    return np.array(feature_values).reshape(1, -1)

def make_prediction(inputs_dict):
    """Make DGF prediction"""
    X = prepare_model_input(inputs_dict, model_features)
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0, 1]
    
    return prediction, probability

# Handle different input modes
if input_mode == "Single Patient":
    st.markdown("### Enter Patient Information")
    
    inputs = get_feature_input()
    
    if st.button("🔮 Predict DGF Risk", type="primary"):
        prediction, probability = make_prediction(inputs)
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("DGF Prediction", "Yes" if prediction == 1 else "No")
        
        with col2:
            st.metric("DGF Probability", f"{probability:.1%}")
        
        with col3:
            risk_level = "High" if probability > 0.3 else "Medium" if probability > 0.15 else "Low"
            st.metric("Risk Level", risk_level)
        
        # Risk interpretation
        if probability > 0.3:
            st.error(f"⚠️ **High Risk**: {probability:.1%} chance of DGF")
            st.write("Consider discussing post-transplant monitoring strategies.")
        elif probability > 0.15:
            st.warning(f"⚠️ **Moderate Risk**: {probability:.1%} chance of DGF")
            st.write("Standard monitoring protocols recommended.")
        else:
            st.success(f"✅ **Low Risk**: {probability:.1%} chance of DGF")
            st.write("Low probability of delayed graft function.")

elif input_mode == "Multiple Patients (CSV Upload)":
    st.markdown("### Upload CSV File")
    
    # Show expected format
    st.markdown("**Expected CSV format:**")
    expected_df = pd.DataFrame({
    'Donor_age': [45], 
    'Donor_final_creatinine': [1.0],
    'Donor_eGFR_CKD_EPI_final_creatinine': [80],
    'Cold_Ischemia_hours': [12.0],
    'Expanded_criteria_donor': [0],
    'KDRI_2024': [1.2]
})
    st.dataframe(expected_df)
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} patients")
            st.dataframe(df.head())
            
            if st.button("Generate Predictions for All Patients"):
                predictions = []
                probabilities = []
                
                for idx, row in df.iterrows():
                    inputs_dict = row.to_dict()
                    pred, prob = make_prediction(inputs_dict)
                    predictions.append(pred)
                    probabilities.append(prob)
                
                results_df = df.copy()
                results_df['DGF_Prediction'] = predictions
                results_df['DGF_Probability'] = probabilities
                results_df['Risk_Level'] = ['High' if p > 0.3 else 'Medium' if p > 0.15 else 'Low' 
                                          for p in probabilities]
                
                st.subheader("Results")
                st.dataframe(results_df)
                
                st.download_button(
                    "📥 Download Results CSV",
                    results_df.to_csv(index=False),
                    "dgf_predictions.csv",
                    "text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

else:  # Manual Entry for Multiple Patients
    st.markdown("### Enter Multiple Patients Manually")
    
    num_patients = st.number_input("Number of patients", min_value=1, max_value=10, value=2)
    
    all_inputs = []
    for i in range(num_patients):
        st.markdown(f"#### Patient {i+1}")
        inputs = get_feature_input(f"- Patient {i+1}", f"_{i}")
        all_inputs.append(inputs)
    
    if st.button("Generate Predictions for All Patients"):
        results = []
        for i, inputs in enumerate(all_inputs):
            pred, prob = make_prediction(inputs)
            results.append({
                'Patient': f"Patient_{i+1}",
                'DGF_Prediction': "Yes" if pred == 1 else "No",
                'DGF_Probability': f"{prob:.1%}",
                'Risk_Level': 'High' if prob > 0.3 else 'Medium' if prob > 0.15 else 'Low'
            })
        
        results_df = pd.DataFrame(results)
        st.subheader("Results Summary")
        st.dataframe(results_df)
        
        st.download_button(
            "📥 Download Results CSV",
            results_df.to_csv(index=False),
            "dgf_predictions.csv",
            "text/csv"
        )

# Sidebar with model info
st.sidebar.markdown("### 📋 Model Information")
st.sidebar.markdown("**Model Type:** Logistic Regression (Donor-Only)")
st.sidebar.markdown("**Target:** Delayed Graft Function (DGF)")
st.sidebar.markdown("**Approach:** Uses only donor characteristics")
st.sidebar.markdown("**Features Used:**")
for feature in model_features:  # Show all features since there are only 6
    st.sidebar.markdown(f"• {feature}")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About DGF")
st.sidebar.markdown("**Delayed Graft Function** occurs when a transplanted kidney doesn't start working immediately, requiring continued dialysis in the first week post-transplant.")